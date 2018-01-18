// @(#)root/tmva $Id$
// Author: Omar Zapata, Thomas James Stevenson and Pourya Vakilipourtakalou
// Modified: Kim Albertsson 2017

/*************************************************************************
 * Copyright (C) 2018, Rene Brun and Fons Rademakers.                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMVA/CrossValidation.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/Config.h"
#include "TMVA/CvSplit.h"
#include "TMVA/DataSet.h"
#include "TMVA/Event.h"
#include "TMVA/MethodBase.h"
#include "TMVA/MethodCrossValidation.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/ResultsClassification.h"
#include "TMVA/ResultsMulticlass.h"
#include "TMVA/ROCCurve.h"
#include "TMVA/tmvaglob.h"
#include "TMVA/Types.h"

#include "TSystem.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TMath.h"

#include <iostream>
#include <memory>

//_______________________________________________________________________
TMVA::CrossValidationResult::CrossValidationResult():fROCCurves(new TMultiGraph())
{
}

//_______________________________________________________________________
TMVA::CrossValidationResult::CrossValidationResult(const CrossValidationResult &obj)
{
   fROCs=obj.fROCs;
   fROCCurves = obj.fROCCurves;
}

//_______________________________________________________________________
TMultiGraph *TMVA::CrossValidationResult::GetROCCurves(Bool_t /*fLegend*/)
{
   return fROCCurves.get();
}

//_______________________________________________________________________
Float_t TMVA::CrossValidationResult::GetROCAverage() const
{
   Float_t avg=0;
   for(auto &roc:fROCs) avg+=roc.second;
   return avg/fROCs.size();
}

//_______________________________________________________________________
Float_t TMVA::CrossValidationResult::GetROCStandardDeviation() const
{
   // NOTE: We are using here the unbiased estimation of the standard deviation.
   Float_t std=0;
   Float_t avg=GetROCAverage();
   for(auto &roc:fROCs) std+=TMath::Power(roc.second-avg, 2);
   return TMath::Sqrt(std/float(fROCs.size()-1.0));
}

//_______________________________________________________________________
void TMVA::CrossValidationResult::Print() const
{
   TMVA::MsgLogger::EnableOutput();
   TMVA::gConfig().SetSilent(kFALSE);

   MsgLogger fLogger("CrossValidation");
   fLogger << kHEADER << " ==== Results ====" << Endl;
   for(auto &item:fROCs)
      fLogger << kINFO << Form("Fold  %i ROC-Int : %.4f",item.first,item.second) << std::endl;

   fLogger << kINFO << "------------------------" << Endl;
   fLogger << kINFO << Form("Average ROC-Int : %.4f",GetROCAverage()) << Endl;
   fLogger << kINFO << Form("Std-Dev ROC-Int : %.4f",GetROCStandardDeviation()) << Endl;

   TMVA::gConfig().SetSilent(kTRUE);
}

//_______________________________________________________________________
TCanvas* TMVA::CrossValidationResult::Draw(const TString name) const
{
   TCanvas *c=new TCanvas(name.Data());
   fROCCurves->Draw("AL");
   fROCCurves->GetXaxis()->SetTitle(" Signal Efficiency ");
   fROCCurves->GetYaxis()->SetTitle(" Background Rejection ");
   Float_t adjust=1+fROCs.size()*0.01;
   c->BuildLegend(0.15,0.15,0.4*adjust,0.5*adjust);
   c->SetTitle("Cross Validation ROC Curves");
   c->Draw();
   return c;
}

/**
* \class TMVA::CrossValidation
* \ingroup TMVA
* \brief

Use html for explicit line breaking<br>
Markdown links? [class reference](#reference)?


~~~{.cpp}
ce->BookMethod(dataloader, options);
ce->Evaluate();
~~~

Cross-evaluation will generate a new training and a test set dynamically from
from `K` folds. These `K` folds are generated by splitting the input training
set. The input test set is currently ignored.

This means that when you specify your DataSet you should include all events
in your training set. One way of doing this would be the following:

~~~{.cpp}
dataloader->AddTree( signalTree, "cls1" );
dataloader->AddTree( background, "cls2" );
dataloader->PrepareTrainingAndTestTree( "", "", "nTest_cls1=1:nTest_cls2=1" );
~~~

## Split Expression
See CVSplit documentation?

*/

////////////////////////////////////////////////////////////////////////////////
///

TMVA::CrossValidation::CrossValidation(TString jobName, TMVA::DataLoader *dataloader, TFile *outputFile,
                                       TString options)
   : TMVA::Envelope(jobName, dataloader, nullptr, options),
     fAnalysisType(Types::kMaxAnalysisType),
     fAnalysisTypeStr("auto"),
     fCorrelations(kFALSE),
     fCvFactoryOptions(""),
     fDrawProgressBar(kFALSE),
     fFoldFileOutput(kFALSE),
     fFoldStatus(kFALSE),
     fJobName(jobName),
     fNumFolds(2),
     fOutputFactoryOptions(""),
     fOutputFile(outputFile),
     fSilent(kFALSE),
     fSplitExprString(""),
     fROC(kTRUE),
     fTransformations(""),
     fVerbose(kFALSE),
     fVerboseLevel(kINFO)
{
   InitOptions();
   ParseOptions();
   CheckForUnusedOptions();

   if (fAnalysisType != Types::kClassification and fAnalysisType != Types::kMulticlass) {
      Log() << kFATAL << "Only binary and multiclass classification supported so far." << Endl;
   }
}

////////////////////////////////////////////////////////////////////////////////
///

TMVA::CrossValidation::CrossValidation(TString jobName, TMVA::DataLoader *dataloader, TString options)
   : CrossValidation(jobName, dataloader, nullptr, options)
{
}

////////////////////////////////////////////////////////////////////////////////
///

TMVA::CrossValidation::~CrossValidation() {}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossValidation::InitOptions()
{
   // Forwarding of Factory options
   DeclareOptionRef(fSilent, "Silent",
                    "Batch mode: boolean silent flag inhibiting any output from TMVA after the creation of the factory "
                    "class object (default: False)");
   DeclareOptionRef(fVerbose, "V", "Verbose flag");
   DeclareOptionRef(fVerboseLevel = TString("Info"), "VerboseLevel", "VerboseLevel (Debug/Verbose/Info)");
   AddPreDefVal(TString("Debug"));
   AddPreDefVal(TString("Verbose"));
   AddPreDefVal(TString("Info"));

   DeclareOptionRef(fTransformations, "Transformations",
                    "List of transformations to test; formatting example: \"Transformations=I;D;P;U;G,D\", for "
                    "identity, decorrelation, PCA, Uniform and Gaussianisation followed by decorrelation "
                    "transformations");

   DeclareOptionRef(fDrawProgressBar, "DrawProgressBar", "Boolean to show draw progress bar");
   DeclareOptionRef(fCorrelations, "Correlations", "Boolean to show correlation in output");
   DeclareOptionRef(fROC, "ROC", "Boolean to show ROC in output");

   TString analysisType("Auto");
   DeclareOptionRef(fAnalysisTypeStr, "AnalysisType",
                    "Set the analysis type (Classification, Regression, Multiclass, Auto) (default: Auto)");
   AddPreDefVal(TString("Classification"));
   AddPreDefVal(TString("Regression"));
   AddPreDefVal(TString("Multiclass"));
   AddPreDefVal(TString("Auto"));

   // Options specific to CE
   DeclareOptionRef(fSplitExprString, "SplitExpr", "The expression used to assign events to folds");
   DeclareOptionRef(fNumFolds, "NumFolds", "Number of folds to generate");

   DeclareOptionRef(fFoldFileOutput, "FoldFileOutput",
                    "If given a TMVA output file will be generated for each fold. Filename will be the same as "
                    "specifed for the combined output with a _foldX suffix. (default: false)");

   DeclareOptionRef(fOutputEnsembling = TString("None"), "OutputEnsembling",
                    "Combines output from contained methods. If None, no combination is performed. (default None)");
   AddPreDefVal(TString("None"));
   AddPreDefVal(TString("Avg"));
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossValidation::ParseOptions()
{
   this->Envelope::ParseOptions();

   // Factory options
   fAnalysisTypeStr.ToLower();
   if (fAnalysisTypeStr == "classification")
      fAnalysisType = Types::kClassification;
   else if (fAnalysisTypeStr == "regression")
      fAnalysisType = Types::kRegression;
   else if (fAnalysisTypeStr == "multiclass")
      fAnalysisType = Types::kMulticlass;
   else if (fAnalysisTypeStr == "auto")
      fAnalysisType = Types::kNoAnalysisType;

   if (fVerbose) {
      fCvFactoryOptions += "V:";
      fOutputFactoryOptions += "V:";
   } else {
      fCvFactoryOptions += "!V:";
      fOutputFactoryOptions += "!V:";
   }

   fCvFactoryOptions += Form("VerboseLevel=%s:", fVerboseLevel.Data());
   fOutputFactoryOptions += Form("VerboseLevel=%s:", fVerboseLevel.Data());

   fCvFactoryOptions += Form("AnalysisType=%s:", fAnalysisTypeStr.Data());
   fOutputFactoryOptions += Form("AnalysisType=%s:", fAnalysisTypeStr.Data());

   if (not fDrawProgressBar) {
      fOutputFactoryOptions += "!DrawProgressBar:";
   }

   if (fTransformations != "") {
      fCvFactoryOptions += Form("Transformations=%s:", fTransformations.Data());
      fOutputFactoryOptions += Form("Transformations=%s:", fTransformations.Data());
   }

   if (fCorrelations) {
      // fCvFactoryOptions += "Correlations:";
      fOutputFactoryOptions += "Correlations:";
   } else {
      // fCvFactoryOptions += "!Correlations:";
      fOutputFactoryOptions += "!Correlations:";
   }

   if (fROC) {
      // fCvFactoryOptions += "ROC:";
      fOutputFactoryOptions += "ROC:";
   } else {
      // fCvFactoryOptions += "!ROC:";
      fOutputFactoryOptions += "!ROC:";
   }

   if (fSilent) {
      // fCvFactoryOptions += Form("Silent:");
      fOutputFactoryOptions += Form("Silent:");
   }

   fCvFactoryOptions += "!Correlations:!ROC:!Color:!DrawProgressBar:Silent";

   // CE specific options
   if (fFoldFileOutput and fOutputFile == nullptr) {
      Log() << kFATAL << "No output file given, cannot generate per fold output." << Endl;
   }

   // Initialisations

   fFoldFactory = std::unique_ptr<TMVA::Factory>(new TMVA::Factory(fJobName, fCvFactoryOptions));

   // The fOutputFactory should always have !ModelPersitence set since we use a custom code path for this.
   //    In this case we create a special method (MethodCrossValidation) that can only be used by
   //    CrossValidation and the Reader.
   if (fOutputFile == nullptr) {
      fFactory = std::unique_ptr<TMVA::Factory>(new TMVA::Factory(fJobName, fOutputFactoryOptions));
   } else {
      fFactory = std::unique_ptr<TMVA::Factory>(new TMVA::Factory(fJobName, fOutputFile, fOutputFactoryOptions));
   }

   fSplit = std::unique_ptr<CvSplitCrossValidation>(new CvSplitCrossValidation(fNumFolds, fSplitExprString));
}

//_______________________________________________________________________
void TMVA::CrossValidation::SetNumFolds(UInt_t i)
{
   if (i != fNumFolds) {
      fNumFolds = i;
      fSplit = std::unique_ptr<CvSplitCrossValidation>(new CvSplitCrossValidation(fNumFolds, fSplitExprString));
      fDataLoader->MakeKFoldDataSet(*fSplit.get());
      fFoldStatus = kTRUE;
   }
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossValidation::SetSplitExpr(TString splitExpr)
{
   if (splitExpr != fSplitExprString) {
      fSplitExprString = splitExpr;
      fSplit = std::unique_ptr<CvSplitCrossValidation>(new CvSplitCrossValidation(fNumFolds, fSplitExprString));
      fDataLoader->MakeKFoldDataSet(*fSplit.get());
      fFoldStatus = kTRUE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Evaluates each fold in turn.
///   - Prepares train and test data sets
///   - Trains method
///   - Evalutes on test set
///   - Stores the evaluation internally
///
/// @param iFold fold to evaluate
///

void TMVA::CrossValidation::ProcessFold(UInt_t iFold, UInt_t iMethod)
{
   TString methodName = fMethods[iMethod].GetValue<TString>("MethodName");
   TString methodTitle = fMethods[iMethod].GetValue<TString>("MethodTitle");
   TString methodOptions = fMethods[iMethod].GetValue<TString>("MethodOptions");

   Log() << kDEBUG << "Fold (" << methodTitle << "): " << iFold << Endl;

   // Get specific fold of dataset and setup method
   TString foldTitle = methodTitle;
   foldTitle += "_fold";
   foldTitle += iFold + 1;

   // Only used if fFoldOutputFile == true
   TFile *foldOutputFile = nullptr;

   if (fFoldFileOutput and fOutputFile != nullptr) {
      TString path = std::string("") + gSystem->DirName(fOutputFile->GetName()) + "/" + foldTitle + ".root";
      std::cout << "PATH: " << path << std::endl;
      foldOutputFile = TFile::Open(path, "RECREATE");
      fFoldFactory = std::unique_ptr<TMVA::Factory>(new TMVA::Factory(fJobName, foldOutputFile, fCvFactoryOptions));
   }

   fDataLoader->PrepareFoldDataSet(*fSplit.get(), iFold, TMVA::Types::kTraining);
   MethodBase *smethod = fFoldFactory->BookMethod(fDataLoader.get(), methodName, foldTitle, methodOptions);

   // Train method (train method and eval train set)
   Event::SetIsTraining(kTRUE);
   smethod->TrainMethod();
   Event::SetIsTraining(kFALSE);

   fFoldFactory->TestAllMethods();
   fFoldFactory->EvaluateAllMethods();

   // Results for aggregation (ROC integral, efficiencies etc.)
   fResults[iMethod].fROCs[iFold] = fFoldFactory->GetROCIntegral(fDataLoader->GetName(), foldTitle);

   TGraph *gr = fFoldFactory->GetROCCurve(fDataLoader->GetName(), foldTitle, true);
   gr->SetLineColor(iFold + 1);
   gr->SetLineWidth(2);
   gr->SetTitle(foldTitle.Data());
   fResults[iMethod].fROCCurves->Add(gr);

   fResults[iMethod].fSigs.push_back(smethod->GetSignificance());
   fResults[iMethod].fSeps.push_back(smethod->GetSeparation());

   if (fAnalysisType == Types::kClassification) {
      Double_t err;
      fResults[iMethod].fEff01s.push_back(smethod->GetEfficiency("Efficiency:0.01", Types::kTesting, err));
      fResults[iMethod].fEff10s.push_back(smethod->GetEfficiency("Efficiency:0.10", Types::kTesting, err));
      fResults[iMethod].fEff30s.push_back(smethod->GetEfficiency("Efficiency:0.30", Types::kTesting, err));
      fResults[iMethod].fEffAreas.push_back(smethod->GetEfficiency("", Types::kTesting, err));
      fResults[iMethod].fTrainEff01s.push_back(smethod->GetTrainingEfficiency("Efficiency:0.01"));
      fResults[iMethod].fTrainEff10s.push_back(smethod->GetTrainingEfficiency("Efficiency:0.10"));
      fResults[iMethod].fTrainEff30s.push_back(smethod->GetTrainingEfficiency("Efficiency:0.30"));
   } else if (fAnalysisType == Types::kMulticlass) {
      // Nothing here for now
   }

   // Per-fold file output
   if (fFoldFileOutput) {
      foldOutputFile->Close();
   }

   // Clean-up for this fold
   {
      smethod->Data()->DeleteResults(foldTitle, Types::kTraining, smethod->GetAnalysisType());
      smethod->Data()->DeleteResults(foldTitle, Types::kTesting, smethod->GetAnalysisType());
   }

   fFoldFactory->DeleteAllMethods();
   fFoldFactory->fMethodsMap.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Does training, test set evaluation and performance evaluation of using
/// cross-evalution.
///

void TMVA::CrossValidation::Evaluate()
{
   // Generate K folds on given dataset
   if (!fFoldStatus) {
      fDataLoader->MakeKFoldDataSet(*fSplit.get());
      fFoldStatus = kTRUE;
   }

   fResults.resize(fMethods.size());
   for (UInt_t iMethod = 0; iMethod < fMethods.size(); iMethod++) {

      TString methodTypeName = fMethods[iMethod].GetValue<TString>("MethodName");
      TString methodTitle = fMethods[iMethod].GetValue<TString>("MethodTitle");

      if (methodTypeName == "") {
         Log() << kFATAL << "No method booked for cross-validation" << Endl;
      }

      TMVA::MsgLogger::EnableOutput();
      Log() << kINFO << "Evaluate method: " << methodTitle << Endl;

      // Process K folds
      for (UInt_t iFold = 0; iFold < fNumFolds; ++iFold) {
         ProcessFold(iFold, iMethod);
      }

      // Serialise the cross evaluated method
      TString options =
         Form("SplitExpr=%s:NumFolds=%i"
              ":EncapsulatedMethodName=%s"
              ":EncapsulatedMethodTypeName=%s"
              ":OutputEnsembling=%s",
              fSplitExprString.Data(), fNumFolds, methodTitle.Data(), methodTypeName.Data(), fOutputEnsembling.Data());

      fFactory->BookMethod(fDataLoader.get(), Types::kCrossValidation, methodTitle, options);
   }

   // Evaluation
   fDataLoader->RecombineKFoldDataSet(*fSplit.get());

   fFactory->TrainAllMethods();
   fFactory->TestAllMethods();
   fFactory->EvaluateAllMethods();

   Log() << kINFO << "Evaluation done." << Endl;
}

//_______________________________________________________________________
const std::vector<TMVA::CrossValidationResult> &TMVA::CrossValidation::GetResults() const
{
   if (fResults.size() == 0)
      Log() << kFATAL << "No cross-validation results available" << Endl;
   return fResults;
}
