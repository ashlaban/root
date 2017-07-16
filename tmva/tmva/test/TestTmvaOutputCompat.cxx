#include "gtest/gtest.h"

#include "TFile.h"
#include "TTree.h"
#include "TKey.h"
#include "TH1.h"
#include "TSystem.h"

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"

// NOTE: GA methods and DNN methods are non-deterministic (at least with standard configuration)
// also TMlpANN

typedef struct {
   TFile * newFile;
   TFile * refFile;

   std::vector<TKey *> newHistKeys;
   std::vector<TKey *> refHistKeys;
} config_t;

// For producing output
void RunClassification();
void RunRegression();
void RunMulticlass();

// Histogram comparison
Bool_t CompareSingleHistogram (TH1 * newHist, TH1 * refHist, const std::string path);
Bool_t CompareAllHistograms (config_t & config);

// Traversing ROOT files
std::vector<TKey *> FindAllKeys(TDirectoryFile * dir);
std::vector<TKey *> FindAllHistogramKeys(TDirectoryFile * dir);

// For debugging
void printClassnameSet(TDirectoryFile * dir);
Bool_t testTH2NumBins();

/* =============================================================================
 * Implementation
 * ========================================================================== */

void RunClassification()
{
   TFile *input(0);
   TString fname = "./tmva_class_example.root";
   if (!gSystem->AccessPathName( fname )) {
      input = TFile::Open( fname ); // check if file in local directory exists
   } else {
      TFile::SetCacheFileDir(".");
      input = TFile::Open("http://root.cern.ch/files/tmva_class_example.root", "CACHEREAD");
   }
   if (!input) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }
   std::cout << "--- TMVAClassification       : Using input file: " << input->GetName() << std::endl;

   TTree *signalTree     = (TTree*)input->Get("TreeS");
   TTree *background     = (TTree*)input->Get("TreeB");

   TString outfileName( "TMVA.root" );
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

   TMVA::Factory * factory = new TMVA::Factory( "TMVAClassification", outputFile, "!V:Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );
   TMVA::DataLoader * dataloader = new TMVA::DataLoader();

   dataloader->AddVariable( "myvar1 := var1+var2", 'F' );
   dataloader->AddVariable( "myvar2 := var1-var2", "Expression 2", "", 'F' );
   dataloader->AddVariable( "var3", "Variable 3", "units", 'F' );
   dataloader->AddVariable( "var4", "Variable 4", "units", 'F' );
   dataloader->AddSpectator( "spec1 := var1*2",  "Spectator 1", "units", 'F' );
   dataloader->AddSpectator( "spec2 := var1*3",  "Spectator 2", "units", 'F' );

   dataloader->AddSignalTree    ( signalTree, 1.0 );
   dataloader->AddBackgroundTree( background, 1.0 );
   dataloader->SetBackgroundWeightExpression( "weight" );
   dataloader->PrepareTrainingAndTestTree( "", "", "nTrain_Signal=1000:nTrain_Background=1000:SplitMode=Random:NormMode=NumEvents:!V" );

   factory->BookMethod( dataloader, TMVA::Types::kCuts, "Cuts","!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart" );
   factory->BookMethod( dataloader, TMVA::Types::kCuts, "CutsD","!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart:VarTransform=Decorrelate" );
   factory->BookMethod( dataloader, TMVA::Types::kCuts, "CutsPCA","!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart:VarTransform=PCA" );
   factory->BookMethod( dataloader, TMVA::Types::kCuts, "CutsSA","!H:!V:FitMethod=SA:EffSel:MaxCalls=150000:KernelTemp=IncAdaptive:InitialTemp=1e+6:MinTemp=1e-6:Eps=1e-10:UseDefaultScale" );
   factory->BookMethod( dataloader, TMVA::Types::kLikelihood, "Likelihood","H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50" );
   factory->BookMethod( dataloader, TMVA::Types::kLikelihood, "LikelihoodD","!H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmooth=5:NAvEvtPerBin=50:VarTransform=Decorrelate" );
   factory->BookMethod( dataloader, TMVA::Types::kLikelihood, "LikelihoodPCA","!H:!V:!TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmooth=5:NAvEvtPerBin=50:VarTransform=PCA" );
   factory->BookMethod( dataloader, TMVA::Types::kLikelihood, "LikelihoodKDE","!H:!V:!TransformOutput:PDFInterpol=KDE:KDEtype=Gauss:KDEiter=Adaptive:KDEFineFactor=0.3:KDEborder=None:NAvEvtPerBin=50" );
   factory->BookMethod( dataloader, TMVA::Types::kLikelihood, "LikelihoodMIX","!H:!V:!TransformOutput:PDFInterpolSig[0]=KDE:PDFInterpolBkg[0]=KDE:PDFInterpolSig[1]=KDE:PDFInterpolBkg[1]=KDE:PDFInterpolSig[2]=Spline2:PDFInterpolBkg[2]=Spline2:PDFInterpolSig[3]=Spline2:PDFInterpolBkg[3]=Spline2:KDEtype=Gauss:KDEiter=Nonadaptive:KDEborder=None:NAvEvtPerBin=50" );
   factory->BookMethod( dataloader, TMVA::Types::kPDERS, "PDERS","!H:!V:NormTree=T:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600" );
   factory->BookMethod( dataloader, TMVA::Types::kPDERS, "PDERSD","!H:!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:VarTransform=Decorrelate" );
   factory->BookMethod( dataloader, TMVA::Types::kPDERS, "PDERSPCA","!H:!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:VarTransform=PCA" );
   factory->BookMethod( dataloader, TMVA::Types::kPDEFoam, "PDEFoam","!H:!V:SigBgSeparate=F:TailCut=0.001:VolFrac=0.0666:nActiveCells=500:nSampl=2000:nBin=5:Nmin=100:Kernel=None:Compress=T" );
   factory->BookMethod( dataloader, TMVA::Types::kPDEFoam, "PDEFoamBoost","!H:!V:Boost_Num=30:Boost_Transform=linear:SigBgSeparate=F:MaxDepth=4:UseYesNoCell=T:DTLogic=MisClassificationError:FillFoamWithOrigWeights=F:TailCut=0:nActiveCells=500:nBin=20:Nmin=400:Kernel=None:Compress=T" );
   factory->BookMethod( dataloader, TMVA::Types::kKNN, "KNN","H:nkNN=20:ScaleFrac=0.8:SigmaFact=1.0:Kernel=Gaus:UseKernel=F:UseWeight=T:!Trim" );
   factory->BookMethod( dataloader, TMVA::Types::kHMatrix, "HMatrix", "!H:!V:VarTransform=None" );
   factory->BookMethod( dataloader, TMVA::Types::kLD, "LD", "H:!V:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10" );
   factory->BookMethod( dataloader, TMVA::Types::kFisher, "Fisher", "H:!V:Fisher:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10" );
   factory->BookMethod( dataloader, TMVA::Types::kFisher, "FisherG", "H:!V:VarTransform=Gauss" );
   factory->BookMethod( dataloader, TMVA::Types::kFisher, "BoostedFisher","H:!V:Boost_Num=20:Boost_Transform=log:Boost_Type=AdaBoost:Boost_AdaBoostBeta=0.2:!Boost_DetailedMonitoring" );
   factory->BookMethod( dataloader, TMVA::Types::kFDA, "FDA_MC","H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MC:SampleSize=100000:Sigma=0.1" );
   factory->BookMethod( dataloader, TMVA::Types::kFDA, "FDA_SA","H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=SA:MaxCalls=15000:KernelTemp=IncAdaptive:InitialTemp=1e+6:MinTemp=1e-6:Eps=1e-10:UseDefaultScale" );
   factory->BookMethod( dataloader, TMVA::Types::kFDA, "FDA_MT","H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=2:UseImprove:UseMinos:SetBatch" );
   factory->BookMethod( dataloader, TMVA::Types::kFDA, "FDA_MCMT","H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MC:Converger=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=0:!UseImprove:!UseMinos:SetBatch:SampleSize=20" );
   factory->BookMethod( dataloader, TMVA::Types::kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:!UseRegulator" );
   factory->BookMethod( dataloader, TMVA::Types::kMLP, "MLPBFGS", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:TrainingMethod=BFGS:!UseRegulator" );
   factory->BookMethod( dataloader, TMVA::Types::kMLP, "MLPBNN", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=60:HiddenLayers=N+5:TestRate=5:TrainingMethod=BFGS:UseRegulator" ); // BFGS training with bayesian regulators
   factory->BookMethod( dataloader, TMVA::Types::kSVM, "SVM", "Gamma=0.25:Tol=0.001:VarTransform=Norm" );
   factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTG","!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=2" );
   factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDT","!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );
   factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTB","!H:!V:NTrees=400:BoostType=Bagging:SeparationType=GiniIndex:nCuts=20" );
   factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTD","!H:!V:NTrees=400:MinNodeSize=5%:MaxDepth=3:BoostType=AdaBoost:SeparationType=GiniIndex:nCuts=20:VarTransform=Decorrelate" );
   factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDTF","!H:!V:NTrees=50:MinNodeSize=2.5%:UseFisherCuts:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20" );
   factory->BookMethod( dataloader, TMVA::Types::kRuleFit, "RuleFit","H:!V:RuleFitModule=RFTMVA:Model=ModRuleLinear:MinImp=0.001:RuleMinDist=0.001:NTrees=20:fEventsMin=0.01:fEventsMax=0.5:GDTau=-1.0:GDTauPrec=0.01:GDStep=0.01:GDNSteps=10000:GDErrScale=1.02" );

   factory->TrainAllMethods();
   factory->TestAllMethods();
   factory->EvaluateAllMethods();
   outputFile->Close();

   delete factory;
   delete dataloader;
}

void RunRegression()
{

}

void RunMulticlass()
{

}

Bool_t CompareSingleHistogram (TH1 * newHist, TH1 * refHist, const std::string path) {
   if (newHist == nullptr or refHist == nullptr) {return kFALSE;}

   if (refHist->GetNcells() != newHist->GetNcells()) {
      std::cerr << "[CompareSingleHistogram] ERROR:" << path << std::endl;
      std::cerr << "[CompareSingleHistogram] Number of bins not equal in histograms." << std::endl;
      std::cerr << "[CompareSingleHistogram] newHist: " << newHist->GetNcells() << " refHist_: " << refHist->GetNcells() << std::endl;
      return kFALSE;
   }

   // NOTE: TH1::GetNcells returns the number of bins regardless of n
   // dimensions and also takes over- and underflow bins into account.
   // TH1::GetBinContent also does this and takes a global bin number as 
   // argument.
   for (Int_t i = 0; i<refHist->GetNcells(); ++i) {
      if (refHist->GetBinContent(i) != newHist->GetBinContent(i)) {
         if ( std::isnan(refHist->GetBinContent(i)) and std::isnan(newHist->GetBinContent(i)) ) {
            continue;
         }
         std::cerr << "[CompareSingleHistogram] ERROR:" << path << std::endl;
         std::cerr << "[CompareSingleHistogram] Mismatch in bin " << i << std::endl;
         std::cerr << "[CompareSingleHistogram] newHist: " << newHist->GetBinContent(i) << " refHist_: " << refHist->GetBinContent(i) << std::endl;
         return kFALSE;
      }
   }

   return kTRUE;
}

Bool_t CompareAllHistograms (config_t & config) {

   if (config.refHistKeys.size() != config.newHistKeys.size()) {
      std::cerr << "[CompareAllHistograms] ERROR: Number of keys not equal. (ref=" << config.refHistKeys.size() << " new=" << config.newHistKeys.size() << ")" << std::endl;
      return kFALSE;
   }

   size_t nKeys = config.refHistKeys.size();
   for (size_t iKey = 0; iKey < nKeys; ++iKey) {
      TKey * newHistKey = config.newHistKeys.at(iKey);
      TKey * refHistKey = config.refHistKeys.at(iKey);

      auto newPath = std::string(newHistKey->GetMotherDir()->GetPathStatic()) + "/" + newHistKey->GetName();
      auto refPath = std::string(refHistKey->GetMotherDir()->GetPathStatic()) + "/" + refHistKey->GetName();

      if (std::string(refHistKey->GetName()) != std::string(newHistKey->GetName())) {
         std::cerr << "[CompareAllHistograms] ERROR: Key name mismatch. (ref=" << refPath << "new=" << newPath << ")" << std::endl;
         return kFALSE;     
      }
      
      auto newHist = (TH1 *)newHistKey->GetMotherDir()->Get(newHistKey->GetName());
      auto refHist = (TH1 *)refHistKey->GetMotherDir()->Get(refHistKey->GetName());
      if (CompareSingleHistogram(newHist, refHist, refPath) == kFALSE) {
         return kFALSE;
      }
   }

   return kTRUE;
}

std::vector<TKey *> FindAllKeys(TDirectoryFile * dir)
{
   if (dir == nullptr) {throw std::runtime_error("[FindAllKeys] ERROR: nullptr input :/");}

   std::vector<TKey *> found_keys;

   // TODO: Fix warning 200:18: warning:
   // loop variable 'item' is always a copy because the range of type 'TList'
   // does not return a reference [-Wrange-loop-analysis]
   for ( auto && item : *(dir->GetListOfKeys()) ) {
      TKey * key = (TKey * )item;
      found_keys.push_back(key);
      
      // std::cout << "name : " << key->GetName() << " -- class: " << key->GetClassName() << std::endl;

      TClass * classObj = TClass::GetClass(key->GetClassName());

      if (key->IsFolder() and classObj->InheritsFrom("TDirectory") ) {
         TDirectoryFile * dirRecurse = static_cast<TDirectoryFile *>(dir->Get(key->GetName()));
         std::vector<TKey *> found_keys_recurse = FindAllKeys(dirRecurse);

         found_keys.insert(found_keys.end(), found_keys_recurse.begin(), found_keys_recurse.end());
      }
   }

   return found_keys;
}

std::vector<TKey *> FindAllHistogramKeys(TDirectoryFile * dir)
{
   std::vector<TKey *> found_histgrams;
   std::vector<TKey *> found_keys = FindAllKeys(dir);

   std::cout << "[FindAllHistogramKeys] Found " << found_keys.size() << " keys." << std::endl;

   for (auto & key : found_keys) {
      // std::cout << "name: " << key->GetName() << " -- class: " << key->GetClassName() << std::endl;
      
      TClass * classObj = TClass::GetClass( key->GetClassName() );
      if (classObj->InheritsFrom("TH1")) {
         found_histgrams.push_back(key);
         // std::string path = std::string(key->GetMotherDir()->GetPath()) + "/";
         // std::cout << "[TH1] name: " << key->GetName() << " -- class: " << key->GetClassName() << std::endl;
      }
   }

   std::cout << "[FindAllHistogramKeys] Found " << found_histgrams.size() << " histograms." << std::endl;

   return found_histgrams;

}

// void printClassnameSet(TDirectoryFile * dir)
// {
//    std::vector<TKey *> keys = FindAllKeys(dir);
//    std::set<std::string> classnames;
   
//    for (auto key : keys) {
//       classnames.insert(key->GetClassName());
//    }

//    std::cout << "[printClassnameSet] Found root objects with classes: ";
//    for (auto name : classnames) {
//       std::cout << name << ", ";
//    }
//    std::cout << std::endl;
// }

// Bool_t testTH2NumBins()
// {
//    // Test that CompareSingleHistogram examines all bins for TH2.

//    // TH2D     (name     , title    , nbinsx, xlow, xup , nbinsy, ylow, yup )
//    TH2D newHist("newHist", "newHist", 10    , 0.0 , 10.0, 10    , 0.0 , 10.0);
//    TH2D refHist("refHist", "refHist", 10    , 0.0 , 10.0, 10    , 0.0 , 10.0);

//    // SetBinContent ensures access is within bounds (clamping to maxbin if above).
//    newHist.SetBinContent(90, 90, 1.0);
//    refHist.SetBinContent(90, 90, 0.0);

//    int isDifferent = CompareSingleHistogram(&newHist, &refHist, "/no/path/");
//    if (not isDifferent) {
//       std::cout << "[testTH2NumBins] SUCCESS: Bin mismatch successfully detected." << std::endl;
//       return kTRUE;
//    } else {
//       std::cout << "[testTH2NumBins] ERROR: Histograms considered equal." << std::endl;
//       return kFALSE;
//    }
   
// }

TEST(TmvaOutputCompat, Classification)
{
   // Right now this test takes ~90 secs on my laptop.
   // It should preferably take ~1 sec. Could we run reuse the 
   // TMVA.root that gets created by TMVAClassification.C?
   
   RunClassification();

   // TODO: Read file from HTTP

   config_t config;
   config.newFile = TFile::Open("TMVA.root");
   config.refFile = TFile::Open("orig/TMVA.root");
   config.newHistKeys = FindAllHistogramKeys(config.newFile);
   config.refHistKeys = FindAllHistogramKeys(config.refFile);

   EXPECT_TRUE(CompareAllHistograms(config));
}

// TEST(TmvaOutputCompat, Regression)
// {
//    RunRegression();

//    config_t config;
//    config.newFile = TFile::Open("TMVAReg.root");
//    config.refFile = TFile::Open("orig/TMVAReg.root");
//    config.newHistKeys = FindAllHistogramKeys(config.newFile);
//    config.refHistKeys = FindAllHistogramKeys(config.refFile);

//    return CompareAllHistograms(config);
// }

// TEST(TmvaOutputCompat, Multiclass)
// {
//    RunMulticlass();

//    config_t config;
//    config.newFile = TFile::Open("TMVAMulticlass.root");
//    config.refFile = TFile::Open("orig/TMVAMulticlass.root");
//    config.newHistKeys = FindAllHistogramKeys(config.newFile);
//    config.refHistKeys = FindAllHistogramKeys(config.refFile);

//    return CompareAllHistograms(config);
// }


////////////////////////////////////////////////////////////////////////////////
/// \param type Should be either of ["Classification", "Regression", "Multiclass"]

// int compare_histograms (TString type = "Classification") {

//    testTH2NumBins();

//    config_t config;

//    if (type == "Classification") {   
//       config.newFile = TFile::Open("TMVA.root");
//       config.refFile = TFile::Open("orig/TMVA.root");
//    } else if (type == "Regression") {
//       config.newFile = TFile::Open("TMVAReg.root");
//       config.refFile = TFile::Open("orig/TMVAReg.root");
//    } else if (type == "Multiclass") {
//       config.newFile = TFile::Open("TMVAMulticlass.root");
//       config.refFile = TFile::Open("orig/TMVAMulticlass.root");
//    } else if (type == "ClassificationApplication") {
//       config.newFile = TFile::Open("TMVApp.root");
//       config.refFile = TFile::Open("orig/TMVApp.root");
//    } else if (type == "RegressionApplication") {
//       config.newFile = TFile::Open("TMVARegApp.root");
//       config.refFile = TFile::Open("orig/TMVARegApp.root");
//    } else if (type == "MulticlassApplication") {
//       config.newFile = TFile::Open("TMVAMulticlassApp.root");
//       config.refFile = TFile::Open("orig/TMVAMulticlassApp.root");
//    }

//    config.newHistKeys = FindAllHistogramKeys(config.newFile);
//    config.refHistKeys = FindAllHistogramKeys(config.refFile);

//    printClassnameSet(config.refFile);
//    printClassnameSet(config.newFile);

//    return CompareAllHistograms(config);
// }
