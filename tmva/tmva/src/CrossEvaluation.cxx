// @(#)root/tmva $Id$
// Author: Kim Albertsson

/*************************************************************************
 * Copyright (C) 2017, Kim Albertsson                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////////
///
//////////////////////////////////////////////////////////////////////////////
#include "TMVA/CrossEvaluation.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/Config.h"
#include "TMVA/CvSplit.h"
#include "TMVA/DataSet.h"
#include "TMVA/Event.h"
#include "TMVA/MethodBase.h"
#include "TMVA/MethodCrossEvaluation.h"
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

/**
* \class TMVA::CrossEvaluation
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
///    TODO: fJobName for fFoldFactory and fFactory ("CrossEvaluation")
///    
///    TODO: Add optional file to fold factory to save output (for debugging at least).
///    

TMVA::CrossEvaluation::CrossEvaluation(TMVA::DataLoader *dataloader, TFile * outputFile, TString options)
   : TMVA::Envelope("CrossEvaluation", dataloader, nullptr, options),
     fAnalysisType(Types::kMaxAnalysisType),
     fAnalysisTypeStr("auto"),
     fCorrelations(kFALSE),
     fFoldStatus(kFALSE),
     fNumFolds(2),
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

TMVA::CrossEvaluation::CrossEvaluation(TMVA::DataLoader *dataloader, TString options)
   : CrossEvaluation(dataloader, nullptr, options)
{}

////////////////////////////////////////////////////////////////////////////////
///

TMVA::CrossEvaluation::~CrossEvaluation()
{}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::InitOptions()
{
   // Forwarding of Factory options
   DeclareOptionRef( fSilent,   "Silent", "Batch mode: boolean silent flag inhibiting any output from TMVA after the creation of the factory class object (default: False)" );
   DeclareOptionRef( fVerbose, "V", "Verbose flag" );
   DeclareOptionRef( fVerboseLevel=TString("Info"), "VerboseLevel", "VerboseLevel (Debug/Verbose/Info)" );
   AddPreDefVal(TString("Debug"));
   AddPreDefVal(TString("Verbose"));
   AddPreDefVal(TString("Info"));
   
   DeclareOptionRef( fTransformations, "Transformations", "List of transformations to test; formatting example: \"Transformations=I;D;P;U;G,D\", for identity, decorrelation, PCA, Uniform and Gaussianisation followed by decorrelation transformations" );

   DeclareOptionRef(fCorrelations, "Correlations", "Boolean to show correlation in output");
   DeclareOptionRef(fROC, "ROC", "Boolean to show ROC in output");

   TString analysisType("Auto");
   DeclareOptionRef( fAnalysisTypeStr, "AnalysisType", "Set the analysis type (Classification, Regression, Multiclass, Auto) (default: Auto)" );
   AddPreDefVal(TString("Classification"));
   AddPreDefVal(TString("Regression"));
   AddPreDefVal(TString("Multiclass"));
   AddPreDefVal(TString("Auto"));

   // Options specific to CE
   DeclareOptionRef( fSplitExprString, "SplitExpr", "The expression used to assign events to folds" );
   DeclareOptionRef( fNumFolds, "NumFolds", "Number of folds to generate" );
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::ParseOptions()
{
   this->Envelope::ParseOptions();

   fAnalysisTypeStr.ToLower();
   if     ( fAnalysisTypeStr == "classification" ) fAnalysisType = Types::kClassification;
   else if( fAnalysisTypeStr == "regression" )     fAnalysisType = Types::kRegression;
   else if( fAnalysisTypeStr == "multiclass" )     fAnalysisType = Types::kMulticlass;
   else if( fAnalysisTypeStr == "auto" )           fAnalysisType = Types::kNoAnalysisType;


   TString fCvFactoryOptions = "";
   TString fOutputFactoryOptions = "";
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

   if (fModelPersistence) {
      fCvFactoryOptions += Form("ModelPersistence:");
   } else {
      fCvFactoryOptions += Form("!ModelPersistence:");
   }

   if (fSilent) {
      // fCvFactoryOptions += Form("Silent:");
      fOutputFactoryOptions += Form("Silent:");
   }

   fFoldFactory = std::unique_ptr<TMVA::Factory>(new TMVA::Factory(
      "CrossEvaluation_internal", fCvFactoryOptions + "!Correlations:!ROC:!Color:!DrawProgressBar:Silent"));

   // The fOutputFactory should always have !ModelPersitence set since we use a custom code path for this.
   //    In this case we create a special method (MethodCrossEvaluation) that can only be used by
   //    CrossEvaluation and the Reader.
   if (fOutputFile == nullptr) {
      fFactory = std::unique_ptr<TMVA::Factory>(new TMVA::Factory("CrossEvaluation",  fOutputFactoryOptions + "!ModelPersistence"));
   } else {
      fFactory = std::unique_ptr<TMVA::Factory>(new TMVA::Factory("CrossEvaluation", fOutputFile,  fOutputFactoryOptions + "!ModelPersistence"));
   }

   fSplit = std::unique_ptr<CvSplitCrossEvaluation>(new CvSplitCrossEvaluation(fNumFolds, fSplitExprString));
   
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::SetNumFolds(UInt_t i)
{
   if (i != fNumFolds) {
      fNumFolds = i;
      fSplit = std::unique_ptr<CvSplitCrossEvaluation>(new CvSplitCrossEvaluation(fNumFolds, fSplitExprString));
      fDataLoader->MakeKFoldDataSet(*fSplit.get());
      fFoldStatus=kTRUE;
   }
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::SetSplitExpr(TString splitExpr)
{
   if (splitExpr != fSplitExprString) {
      fSplitExprString = splitExpr;
      fSplit = std::unique_ptr<CvSplitCrossEvaluation>(new CvSplitCrossEvaluation(fNumFolds, fSplitExprString));
      fDataLoader->MakeKFoldDataSet(*fSplit.get());
      fFoldStatus=kTRUE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Extract test set results from DataSet for given method and store this
/// internally.
///
/// @param smethod method to which extract results for
///

void TMVA::CrossEvaluation::StoreFoldResults(MethodBase * smethod) {
      DataSet * ds = fDataLoader->GetDataSetInfo().GetDataSet();
      ResultsClassification * resultTestSet =
         dynamic_cast<ResultsClassification *>( ds->GetResults(smethod->GetName(), 
                                                Types::kTesting,
                                                smethod->GetAnalysisType()));

      EventCollection_t evCollection = ds->GetEventCollection(Types::kTesting);

      fOutputsPerFold.push_back( *resultTestSet->GetValueVector()      );
      fClassesPerFold.push_back( *resultTestSet->GetValueVectorTypes() );
}

////////////////////////////////////////////////////////////////////////////////
/// Clears the internal caches of fold results.
///

void TMVA::CrossEvaluation::ClearFoldResultsCache() {
      fOutputsPerFold.clear();
      fClassesPerFold.clear();
      fOutputsPerFoldMulticlass.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Assembles fold results stored by `StoreResults` and injects them into the
/// DataSet connecting it to the given method. Both the test and train results
/// are injected.
///
/// @note The train results are copies of the test ones. This is subject to
/// change in future revisions.
///
/// @param smethod results are generated for this method
///

void TMVA::CrossEvaluation::MergeFoldResults(MethodBase * smethod)
{
   DataSet * ds = fDataLoader->GetDataSetInfo().GetDataSet();
   EventOutputs_t outputs;
   EventTypes_t classes;
   for(UInt_t iFold = 0; iFold < fNumFolds; ++iFold) {
      outputs.insert(outputs.end(), fOutputsPerFold.at(iFold).begin(), fOutputsPerFold.at(iFold).end());
      classes.insert(classes.end(), fClassesPerFold.at(iFold).begin(), fClassesPerFold.at(iFold).end());
   }

   TString              methodName   = smethod->GetName();
   Types::EAnalysisType analysisType = smethod->GetAnalysisType();

   ResultsClassification * metaResults;

   // For now this is a copy of the testing set. We might want to inject training data here.
   metaResults = dynamic_cast<ResultsClassification *>(ds->GetResults(methodName, Types::kTraining, analysisType));
   metaResults->GetValueVector()->insert(metaResults->GetValueVector()->begin(), outputs.begin(), outputs.end());
   metaResults->GetValueVectorTypes()->insert(metaResults->GetValueVectorTypes()->begin(), classes.begin(), classes.end());

   metaResults = dynamic_cast<ResultsClassification *>(ds->GetResults(methodName, Types::kTesting, analysisType));
   metaResults->GetValueVector()->insert(metaResults->GetValueVector()->begin(), outputs.begin(), outputs.end());
   metaResults->GetValueVectorTypes()->insert(metaResults->GetValueVectorTypes()->begin(), classes.begin(), classes.end());
}

////////////////////////////////////////////////////////////////////////////////
/// Extract test set results from DataSet for given method and store this
/// internally.
///
/// @param smethod method to which extract results for
///

void TMVA::CrossEvaluation::StoreFoldResultsMulticlass(MethodBase * smethod)
{
      DataSet * ds = fDataLoader->GetDataSetInfo().GetDataSet();
      ResultsMulticlass * resultTestSet =
         dynamic_cast<ResultsMulticlass *>( ds->GetResults(smethod->GetName(),
                                            Types::kTesting,
                                            smethod->GetAnalysisType()));

      fOutputsPerFoldMulticlass.push_back( *resultTestSet->GetValueVector());
}

////////////////////////////////////////////////////////////////////////////////
/// Assembles fold results stored by `StoreResults` and injects them into the
/// DataSet connecting it to the given method. Both the test and train results
/// are injected.
///
/// @note The train results are copies of the test ones. This is subject to
/// change in future revisions.
///
/// @param smethod results are generated for this method
///

void TMVA::CrossEvaluation::MergeFoldResultsMulticlass(MethodBase * smethod)
{
   DataSet * ds = fDataLoader->GetDataSetInfo().GetDataSet();
   EventOutputsMulticlass_t outputs;
   for(UInt_t iFold = 0; iFold < fNumFolds; ++iFold) {
      outputs.insert(outputs.end(), fOutputsPerFoldMulticlass.at(iFold).begin(), fOutputsPerFoldMulticlass.at(iFold).end());
   }

   TString              methodName   = smethod->GetName();
   Types::EAnalysisType analysisType = smethod->GetAnalysisType();

   ResultsMulticlass * metaResults;

   // For now this is a copy of the testing set. We might want to inject training data here.
   metaResults = dynamic_cast<ResultsMulticlass *>(ds->GetResults(methodName, Types::kTraining, analysisType));
   metaResults->GetValueVector()->insert(metaResults->GetValueVector()->begin(), outputs.begin(), outputs.end());

   metaResults = dynamic_cast<ResultsMulticlass *>(ds->GetResults(methodName, Types::kTesting, analysisType));
   metaResults->GetValueVector()->insert(metaResults->GetValueVector()->begin(), outputs.begin(), outputs.end());
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

void TMVA::CrossEvaluation::ProcessFold(UInt_t iFold)
{
   TString methodName    = fMethod.GetValue<TString>("MethodName");
   TString methodTitle   = fMethod.GetValue<TString>("MethodTitle");
   TString methodOptions = fMethod.GetValue<TString>("MethodOptions");

   Log() << kDEBUG << "Fold (" << methodTitle << "): " << iFold << Endl;

   // Get specific fold of dataset and setup method
   TString foldTitle = methodTitle;
   foldTitle += "_fold";
   foldTitle += iFold+1;


   fDataLoader->PrepareFoldDataSet(*fSplit.get(), iFold, TMVA::Types::kTraining);
   MethodBase* smethod = fFoldFactory->BookMethod(fDataLoader.get(), methodName, foldTitle, methodOptions);

   // Train method (train method and eval train set)
   Event::SetIsTraining(kTRUE);
   smethod->TrainMethod();

   // Test method (evaluate the test set)
   Event::SetIsTraining(kFALSE);
   smethod->AddOutput(Types::kTesting, smethod->GetAnalysisType());

   switch (fAnalysisType) {
      case Types::kClassification: StoreFoldResults(smethod); break;
      case Types::kMulticlass    : StoreFoldResultsMulticlass(smethod); break;
      default:
         Log() << kFATAL << "CrossEvaluation currently supports only classification and multiclass classification." << Endl;
         break;
   }

   // Clean-up for this fold
   smethod->Data()->DeleteResults(foldTitle, Types::kTesting, smethod->GetAnalysisType());
   smethod->Data()->DeleteResults(foldTitle, Types::kTraining, smethod->GetAnalysisType());
   fFoldFactory->DeleteAllMethods();
   fFoldFactory->fMethodsMap.clear();
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::MergeFolds()
{

   TString methodName    = fMethod.GetValue<TString>("MethodName");
   TString methodTitle   = fMethod.GetValue<TString>("MethodTitle");
   TString methodOptions = fMethod.GetValue<TString>("MethodOptions");

   fFactory->BookMethod(fDataLoader.get(), methodName, methodTitle, methodOptions);

   MethodBase * smethod = dynamic_cast<MethodBase *>(fFactory->GetMethod(fDataLoader->GetName(), methodTitle));

   // Write data such as VariableTransformations to output file.
   if (fOutputFile != nullptr) {
      fFactory->WriteDataInformation(smethod->DataInfo());
   }

   // Merge results from the folds into a single result
   switch (fAnalysisType) {
      case Types::kClassification: MergeFoldResults(smethod); break;
      case Types::kMulticlass    : MergeFoldResultsMulticlass(smethod); break;
      default:
         Log() << kFATAL << "CrossEvaluation currently supports only classification and multiclass classification." << Endl;
         break;
   }

   // Merge inputs 
   fDataLoader->RecombineKFoldDataSet( *fSplit.get() );
}

////////////////////////////////////////////////////////////////////////////////
/// Does training, test set evaluation and performance evaluation of using
/// cross-evalution.
///

void TMVA::CrossEvaluation::Evaluate()
{
   TString methodName  = fMethod.GetValue<TString>("MethodName");
   TString methodTitle = fMethod.GetValue<TString>("MethodTitle");
   if(methodName == "") Log() << kFATAL << "No method booked for cross-validation" << Endl;

   TMVA::MsgLogger::EnableOutput();
   Log() << kINFO << "Evaluate method: " << methodTitle << Endl;

   // Generate K folds on given dataset
   if(!fFoldStatus){
       fDataLoader->MakeKFoldDataSet(*fSplit.get());
       fFoldStatus=kTRUE;
   }

   // Process K folds
   for(UInt_t iFold = 0; iFold < fNumFolds; ++iFold) {
      ProcessFold(iFold);
   }

   // Merge and inject the results into DataSet
   MergeFolds();
   ClearFoldResultsCache();

   // Run produce final output (e.g. file)
   fFactory->EvaluateAllMethods();

   // Serialise the cross evaluated method
   if (fModelPersistence) {
      // Create new MethodCrossEvaluation
      TString methodCrossEvaluationName = Types::Instance().GetMethodName( Types::kCrossEvaluation );
      IMethod * im = ClassifierFactory::Instance().Create( methodCrossEvaluationName.Data(),
                                                           "", // jobname
                                                           "CrossEvaluation_"+methodTitle,   // title
                                                           fDataLoader->GetDataSetInfo(), // dsi
                                                           "" // options
                                                         ); 

      // Serialise it
      MethodBase * method = dynamic_cast<MethodBase *>(im);

      // Taken directly from what is done in Factory::BookMethod
      TString fFileDir = TString(fDataLoader->GetName()) + "/" + gConfig().GetIONames().fWeightFileDir;
      method->SetWeightFileDir(fFileDir);
      method->SetModelPersistence(fModelPersistence);
      method->SetAnalysisType(fAnalysisType);
      method->SetupMethod();
      method->ParseOptions();
      method->ProcessSetup();
      // method->SetFile(fgTargetFile);
      // method->SetSilentFile(IsSilentFile());

      // check-for-unused-options is performed; may be overridden by derived classes
      method->CheckSetup();

      // Pass info about the correct method name (method_title_base + foldNum)
      // Pass info about the number of folds
      // TODO: Parameterise the internal jobname
      MethodCrossEvaluation * method_ce = dynamic_cast<MethodCrossEvaluation *>(method);
      method_ce->fEncapsulatedMethodName     = "CrossEvaluation_internal_" + methodTitle;
      method_ce->fEncapsulatedMethodTypeName = methodName;
      method_ce->fNumFolds                   = fNumFolds;
      method_ce->fSplitExprString            = fSplitExprString;

      method->WriteStateToFile();
      // Not supported by MethodCrossEvaluation yet
      // if (fAnalysisType != Types::kRegression) { smethod->MakeClass(); }
   }

   Log() << kINFO << "Evaluation done." << Endl;
}
