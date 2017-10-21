// @(#)root/tmva $Id$
// Author: Kim Albertsson

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodCrossEvaluation                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodCrossEvaluation
#define ROOT_TMVA_MethodCrossEvaluation

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodCrossEvaluation                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMVA/CvSplit.h"
#include "TMVA/DataSetInfo.h"
#include "TMVA/MethodBase.h"

#include "TString.h"

#include <iostream>
#include <memory>

namespace TMVA {

   class Ranking;

   // Looks for serialised methods of the form methodTitle + "_fold" + iFold;
   class MethodCrossEvaluation : public MethodBase {

   public:
      // constructor for training and reading
      MethodCrossEvaluation( const TString&     jobName,
                             const TString&     methodTitle,
                                   DataSetInfo& theData,
                             const TString&     theOption = "");

      // constructor for calculating BDT-MVA using previously generatad decision trees
      MethodCrossEvaluation(       DataSetInfo& theData,
                             const TString&     theWeightFile);

      virtual ~MethodCrossEvaluation( void );

      // optimize tuning parameters
      // virtual std::map<TString,Double_t> OptimizeTuningParameters(TString fomType="ROCIntegral", TString fitType="FitGA");
      // virtual void SetTuneParameters(std::map<TString,Double_t> tuneParameters);

      // training method
      void Train( void );

      // revoke training
      void Reset( void );

      using MethodBase::ReadWeightsFromStream;

      // write weights to file
      void AddWeightsXMLTo( void* parent ) const;

      // read weights from file
      void ReadWeightsFromStream( std::istream& istr );
      void ReadWeightsFromXML(void* parent);

      // write method specific histos to target file
      void WriteMonitoringHistosToFile( void ) const;

      // calculate the MVA value
      Double_t GetMvaValue( Double_t* err = 0, Double_t* errUpper = 0);
      const std::vector<Float_t>& GetMulticlassValues();
      const std::vector<Float_t>& GetRegressionValues();

      // the option handling methods
      void DeclareOptions();
      void ProcessOptions();

      // make ROOT-independent C++ class for classifier response (classifier-specific implementation)
      void MakeClassSpecific( std::ostream&, const TString& ) const;
      void MakeClassSpecificHeader( std::ostream&, const TString& ) const;

      void GetHelpMessage() const;

      const Ranking * CreateRanking();
      Bool_t HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets );

   protected:
      void Init( void );
      void DeclareCompatibilityOptions();

   private:
      MethodBase * InstantiateMethodFromXML(TString methodTypeName, TString weightfile) const;

   public:
      // TODO: Only public until proper getter and setters are implemented
      // TODO: Add setter both for EMVA and String Typename directly.
      TString fEncapsulatedMethodName;
      TString fEncapsulatedMethodTypeName;
      UInt_t  fNumFolds;
      TString fOutputEnsembling;
      
      TString fSplitExprString;
      std::unique_ptr<CvSplitCrossEvaluationExpr> fSplitExpr;

   private:
      // MethodBase::fFileDir gives path to weightfiles

      std::vector<Float_t> fMulticlassValues;

      std::vector<MethodBase *> fEncapsulatedMethods;

      // Temporary holder of data while GetMulticlassValues and GetRegressionValues
      // are not implemented.
      std::vector<Float_t> fNotImplementedRetValVec;

      // debugging flags
      static const Int_t fgDebugLevel;     // debug level determining some printout/control plots etc.

      // for backward compatibility

      ClassDef(MethodCrossEvaluation, 0);
   };

} // namespace TMVA

#endif