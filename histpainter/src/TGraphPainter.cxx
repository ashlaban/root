// @(#)root/histpainter:$Name:  $:$Id: TGraphPainter.cxx,v 1.00
// Author: Olivier Couet

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TGraphPainter.h"
#include "TMath.h"
#include "TPolyLine.h"
#include "TPolyMarker.h"
#include "TVirtualPad.h"
#include "TView.h"
#include "THLimitsFinder.h"
#include "TStyle.h"
#include "Hoption.h"

R__EXTERN TH1  *gCurrentHist;
R__EXTERN Hoption_t Hoption;

ClassImp(TGraphPainter)


//______________________________________________________________________________
//
// TGraphPainter paints a TGraphDelaunay
//


//______________________________________________________________________________
TGraphPainter::TGraphPainter()
{
   // TGraphPainter default constructor

   fDelaunay = 0;
}


//______________________________________________________________________________
TGraphPainter::TGraphPainter(TGraphDelaunay *gd)
{
   // TGraphPainter constructor

   fDelaunay = gd;
   fGraph2D  = fDelaunay->GetGraph2D();
   fNpoints  = fGraph2D->GetN();
   fX        = fGraph2D->GetX();
   fY        = fGraph2D->GetY();
   fZ        = fGraph2D->GetZ();
   fNdt      = 0;
   fXN       = 0;
   fYN       = 0;
   fXNmin    = 0;
   fXNmax    = 0;
   fYNmin    = 0;
   fYNmax    = 0;
   fPTried   = 0;
   fNTried   = 0;
   fMTried   = 0;
}


//______________________________________________________________________________
TGraphPainter::~TGraphPainter()
{
   // TGraphPainter destructor.
}


//______________________________________________________________________________
void TGraphPainter::Paint(Option_t *option)
{
   // Paint a TGraphDelaunay according to the value of "option":
   //
   //   "TRI"  : The Delaunay triangles are drawn using filled area.
   //            An hidden surface drawing technique is used. The surface is  
   //            painted with the current fill area color. The edges of each
   //            triangles are painted with the current line color.
   //   "TRIW" : The Delaunay triangles are drawn as wire frame
   //   "TRI1" : The Delaunay triangles are painted with color levels. The edges
   //            of each triangles are painted with the current line color.
   //   "TRI2" : the Delaunay triangles are painted with color levels.
   //   "P"    : Draw a marker at each vertex
   //   "P0"   : Draw a circle at each vertex. Each circle background is white.
   //   "CONT" : Draw contours

   TString opt = option;
   opt.ToLower();
   Bool_t triangles = opt.Contains("tri")  ||
                      opt.Contains("tri1") ||
                      opt.Contains("tri2"); 
   if (opt.Contains("tri0")) triangles = kFALSE;

   Bool_t markers   = opt.Contains("p") && !triangles;
   Bool_t contour   = opt.Contains("cont");

   fGraph2D->TAttLine::Modify();
   fGraph2D->TAttFill::Modify();
   fGraph2D->TAttMarker::Modify();

   // Compute minimums and maximums
   TAxis *xaxis = gCurrentHist->GetXaxis();
   Int_t first = xaxis->GetFirst();
   fXmin = xaxis->GetBinLowEdge(first);
   if (Hoption.Logx && fXmin <= 0) fXmin = xaxis->GetBinUpEdge(xaxis->FindFixBin(0.01*xaxis->GetBinWidth(first)));
   fXmax = xaxis->GetBinUpEdge(xaxis->GetLast());
   TAxis *yaxis = gCurrentHist->GetYaxis();
   first = yaxis->GetFirst();
   fYmin = yaxis->GetBinLowEdge(first);
   if (Hoption.Logy && fYmin <= 0) fYmin = yaxis->GetBinUpEdge(yaxis->FindFixBin(0.01*yaxis->GetBinWidth(first)));
   fYmax = yaxis->GetBinUpEdge(yaxis->GetLast());
   fZmax = gCurrentHist->GetMaximum();
   fZmin = gCurrentHist->GetMinimum();
   if (Hoption.Logz && fZmin <= 0) fZmin = TMath::Min((Double_t)1, (Double_t)0.001*gCurrentHist->GetMaximum());

   if (triangles) PaintTriangles(option);
   if (markers)   PaintPolyMarker(option);
   if (contour)   PaintContour(option);
}


//______________________________________________________________________________
void TGraphPainter::PaintContour(Option_t * /*option*/)
{
   // Paints the 2D graph as a contour plot

   Int_t ndivz   = gCurrentHist->GetContour();
   if (ndivz == 0 ) {
      ndivz = gStyle->GetNumberContours();
      gCurrentHist->SetContour(ndivz);
   }

   Int_t ncolors = gStyle->GetNumberOfColors();
   Int_t theColor;
   TList *l;
   TGraph *g;
   TObject *obj;
   Double_t c;

   fDelaunay->FindAllTriangles();

   for (Int_t k=0; k<ndivz; k++) {
      c = gCurrentHist->GetContourLevelPad(k);
      l = fDelaunay->GetContourList(c);
      TIter next(l);   
      while ((obj = next())) {
         if(obj->InheritsFrom(TGraph::Class()) ) {
            g=(TGraph*)obj;
            theColor = Int_t((k+0.99)*Float_t(ncolors)/Float_t(ndivz));
            g->SetLineColor(gStyle->GetColorPalette(theColor));
            g->Paint("l");
         }
      }
   }
}


//______________________________________________________________________________
void TGraphPainter::PaintLevels(Int_t *T,Double_t *x, Double_t *y,
                           Int_t nblev, Double_t *glev)
{
   // Paints one triangle.
   // nblev  = 0 : paint the color levels
   // nblev != 0 : paint the grid

   Int_t i, FC, ncolors, theColor0, theColor2;
   
   Int_t P0=T[0]-1;
   Int_t P1=T[1]-1;
   Int_t P2=T[2]-1;
   Double_t xl[2],yl[2];
   Double_t Zl, R21, R20, R10;
   Double_t X0 = x[0]  , X2 = x[0];
   Double_t Y0 = y[0]  , Y2 = y[0];
   Double_t Z0 = fZ[P0], Z2 = fZ[P0];

   // Order along Z axis the points (Xi,Yi,Zi) where "i" belongs to {0,1,2}
   // After this Z0 < Z1 < Z2
   Int_t I0=0, I1=0, I2=0;
   if (fZ[P1]<=Z0) {Z0=fZ[P1]; X0=x[1]; Y0=y[1]; I0=1;}
   if (fZ[P1]>Z2)  {Z2=fZ[P1]; X2=x[1]; Y2=y[1]; I2=1;}
   if (fZ[P2]<=Z0) {Z0=fZ[P2]; X0=x[2]; Y0=y[2]; I0=2;}
   if (fZ[P2]>Z2)  {Z2=fZ[P2]; X2=x[2]; Y2=y[2]; I2=2;}
   I1 = 3-I2-I0;
   Double_t X1 = x[I1];
   Double_t Y1 = y[I1];
   Double_t Z1 = fZ[T[I1]-1];

   // Zi  = Z values of the stripe number i
   // Zip = Previous Zi 
   Double_t Zi=0, Zip=0;

   if (nblev <= 0) {
      // Paint the colors levels

      // Compute the color associated to Z0 (theColor0) and Z2 (theColor2)
      ncolors   = gStyle->GetNumberOfColors();
      theColor0 = (Int_t)( ((Z0-fZmin)/(fZmax-fZmin))*(ncolors-1) );
      theColor2 = (Int_t)( ((Z2-fZmin)/(fZmax-fZmin))*(ncolors-1) );

      // The stripes drawn to fill the triangles may have up to 5 points
      Double_t xp[5], yp[5];

      // Rl = Ratio between Z0 and Z2 (long) 
      // Rs = Ratio between Z0 and Z1 or Z1 and Z2 (short) 
      Double_t Rl,Rs;

      // Ci = Color of the stripe number i
      // npf = number of point needed to draw the current stripe
      Int_t Ci,npf;

      FC = fGraph2D->GetFillColor();

      // If the Z0's color and Z2's colors are the same, the whole triangle
      // can be painted in one go.
      if(theColor0 == theColor2) {
         fGraph2D->SetFillColor(gStyle->GetColorPalette(theColor0));
         fGraph2D->TAttFill::Modify();
         gPad->PaintFillArea(3,x,y);

      // The triangle must be painted with several colors
      } else {
         for(Ci=theColor0; Ci<=theColor2; Ci++) {
            fGraph2D->SetFillColor(gStyle->GetColorPalette(Ci));
            fGraph2D->TAttFill::Modify();
            if (Ci==theColor0) {
               Zi    = (((Ci+1)*(fZmax-fZmin))/(ncolors-1))+fZmin;
               xp[0] = X0;
               yp[0] = Y0;
               Rl    = (Zi-Z0)/(Z2-Z0);
               xp[1] = Rl*(X2-X0)+X0;
               yp[1] = Rl*(Y2-Y0)+Y0;
               if (Zi>=Z1 || Z0==Z1) {
                  Rs    = (Zi-Z1)/(Z2-Z1);
                  xp[2] = Rs*(X2-X1)+X1;
                  yp[2] = Rs*(Y2-Y1)+Y1;
                  xp[3] = X1;
                  yp[3] = Y1;
                  npf   = 4;
                } else {
                  Rs    = (Zi-Z0)/(Z1-Z0);
                  xp[2] = Rs*(X1-X0)+X0;
                  yp[2] = Rs*(Y1-Y0)+Y0;
                  npf   = 3;
               }
            } else if (Ci==theColor2) {
               xp[0] = xp[1];
               yp[0] = yp[1];
               xp[1] = X2;
               yp[1] = Y2;
               if (Zi<Z1 || Z2==Z1) {
                  xp[3] = xp[2];
                  yp[3] = yp[2];
                  xp[2] = X1;
                  yp[2] = Y1;
                  npf   = 4;
               } else {
                  npf   = 3;
               }
            } else {
               Zi    = (((Ci+1)*(fZmax-fZmin))/(ncolors-1))+fZmin;
               xp[0] = xp[1];
               yp[0] = yp[1];
               Rl    = (Zi-Z0)/(Z2-Z0);
               xp[1] = Rl*(X2-X0)+X0;
               yp[1] = Rl*(Y2-Y0)+Y0;
               if ( Zi>=Z1 && Zip<=Z1) {
                  xp[3] = X1;
                  yp[3] = Y1;
                  xp[4] = xp[2];
                  yp[4] = yp[2];
                  npf   = 5;
               } else {
                  xp[3] = xp[2];
                  yp[3] = yp[2];
                  npf   = 4;
               }
               if (Zi<Z1) {
                  Rs    = (Zi-Z0)/(Z1-Z0);
                  xp[2] = Rs*(X1-X0)+X0;
                  yp[2] = Rs*(Y1-Y0)+Y0;
               } else {
                  Rs    = (Zi-Z1)/(Z2-Z1);
                  xp[2] = Rs*(X2-X1)+X1;
                  yp[2] = Rs*(Y2-Y1)+Y1;
               }
            }
            Zip = Zi;
            // Paint a stripe
            gPad->PaintFillArea(npf,xp,yp);
         }
      }
      fGraph2D->SetFillColor(FC);
      fGraph2D->TAttFill::Modify();

   } else {
      // Paint the grid levels
      fGraph2D->SetLineStyle(3);
      fGraph2D->TAttLine::Modify();
      for(i=0; i<nblev; i++){
         Zl=glev[i];
         if(Zl >= Z0 && Zl <=Z2) {
            R21=(Zl-Z1)/(Z2-Z1);
            R20=(Zl-Z0)/(Z2-Z0);
            R10=(Zl-Z0)/(Z1-Z0);
            xl[0]=R20*(X2-X0)+X0;
            yl[0]=R20*(Y2-Y0)+Y0;
            if(Zl >= Z1 && Zl <=Z2) {
               xl[1]=R21*(X2-X1)+X1;
               yl[1]=R21*(Y2-Y1)+Y1;
            } else {
               xl[1]=R10*(X1-X0)+X0;
               yl[1]=R10*(Y1-Y0)+Y0;
            }
            gPad->PaintPolyLine(2,xl,yl);
         }
      }
      fGraph2D->SetLineStyle(1);
      fGraph2D->TAttLine::Modify();
   }
}


//______________________________________________________________________________
void TGraphPainter::PaintPolyMarker(Option_t *option)
{
   // Paints the 2D graph as PaintPolyMarker

   Double_t temp1[3],temp2[3];

   TView *view = gPad->GetView();
   if (!view) {
      Error("PaintTriangles", "No TView in current pad");
      return;
   }

   TString opt = option;
   opt.ToLower();
   Bool_t markers0  = opt.Contains("p0");

   Double_t *xm = new Double_t[fNpoints]; 
   Double_t *ym = new Double_t[fNpoints];
   Int_t    npd = 0;
   for (Int_t IT=0; IT<fNpoints; IT++) {
      if(fX[IT] < fXmin || fX[IT] > fXmax) continue;
      if(fY[IT] < fYmin || fY[IT] > fYmax) continue;
      npd++;
      temp1[0] = fX[IT];
      temp1[1] = fY[IT];
      temp1[2] = fZ[IT];
      temp1[0] = TMath::Max(temp1[0],fXmin);
      temp1[1] = TMath::Max(temp1[1],fYmin);
      temp1[2] = TMath::Max(temp1[2],fZmin);
      temp1[2] = TMath::Min(temp1[2],fZmax);
      if (Hoption.Logx) temp1[0] = TMath::Log10(temp1[0]);
      if (Hoption.Logy) temp1[1] = TMath::Log10(temp1[1]);
      if (Hoption.Logz) temp1[2] = TMath::Log10(temp1[2]);
      view->WCtoNDC(temp1, &temp2[0]);
      xm[IT] = temp2[0];
      ym[IT] = temp2[1];
   }
   if (markers0) {
      PaintPolyMarker0(npd,xm,ym);
   } else {
      fGraph2D->SetMarkerStyle(fGraph2D->GetMarkerStyle());
      fGraph2D->SetMarkerSize(fGraph2D->GetMarkerSize());
      fGraph2D->SetMarkerColor(fGraph2D->GetMarkerColor());
      fGraph2D->TAttMarker::Modify();
      gPad->PaintPolyMarker(npd,xm,ym);
   }
   delete [] xm;
   delete [] ym;
}


//______________________________________________________________________________
void TGraphPainter::PaintPolyMarker0(Int_t n, Double_t *x, Double_t *y)
{
   // Paints a circle at each vertex. Each circle background is white. 

   fGraph2D->SetMarkerSize(fGraph2D->GetMarkerSize());
   Int_t MC = fGraph2D->GetMarkerColor();
   for (Int_t i=0; i<n; i++) {
      fGraph2D->SetMarkerStyle(20);
      fGraph2D->SetMarkerColor(0);
      fGraph2D->TAttMarker::Modify();
      gPad->PaintPolyMarker(1,&x[i],&y[i]);
      fGraph2D->SetMarkerStyle(24);
      fGraph2D->SetMarkerColor(MC);
      fGraph2D->TAttMarker::Modify();
      gPad->PaintPolyMarker(1,&x[i],&y[i]);
   }
}


//______________________________________________________________________________
void TGraphPainter::PaintTriangles(Option_t *option)
{
   // Paints the 2D graph as triangles

   Double_t x[4], y[4], temp1[3],temp2[3];
   Int_t IT,T[3];
   Int_t *order = 0;
   Double_t *dist = 0;

   TView *view = gPad->GetView();
   if (!view) {
      Error("PaintTriangles", "No TView in current pad");
      return;
   }

   TString opt = option;
   opt.ToLower();
   Bool_t tri1      = opt.Contains("tri1"); 
   Bool_t tri2      = opt.Contains("tri2"); 
   Bool_t markers   = opt.Contains("p");
   Bool_t markers0  = opt.Contains("p0");
   Bool_t wire      = opt.Contains("w");

   // Define the grid levels drawn on the triangles.
   // The grid levels are aligned on the Z axis' main tick marks.
   Int_t nblev=0;
   Double_t *glev=0;
   if (!tri1 && !tri2 && !wire) {
      Int_t ndivz = gCurrentHist->GetZaxis()->GetNdivisions()%100;
      Int_t nbins;
      Double_t BinLow, BinHigh, BinWidth;

      // Find the main tick marks positions.
      Double_t *r0 = view->GetRmin();
      Double_t *r1 = view->GetRmax();

      if (ndivz > 0) {
         THLimitsFinder::Optimize(r0[2], r1[2], ndivz,
                                  BinLow, BinHigh, nbins, BinWidth, " ");
      } else {
         nbins = TMath::Abs(ndivz);
         BinLow = r0[2];
         BinHigh = r1[2];
         BinWidth = (BinHigh-BinLow)/nbins;
      }
      // Define the grid levels
      nblev = nbins+1;
      glev = new Double_t[nblev];
      for (Int_t i = 0; i < nblev; ++i) glev[i] = BinLow+i*BinWidth;
   }

   // For each triangle, compute the distance between the triangle centre
   // and the back planes. Then these distances are sorted in order to draw
   // the triangles from back to front. 
   fDelaunay->FindAllTriangles();
   fNdt    = fDelaunay->GetNdt();
   fXN     = fDelaunay->GetXN();
   fYN     = fDelaunay->GetYN();
   fXNmin  = fDelaunay->GetXNmin();
   fXNmax  = fDelaunay->GetXNmax();
   fYNmin  = fDelaunay->GetYNmin();
   fYNmax  = fDelaunay->GetYNmax();
   fPTried = fDelaunay->GetPTried();
   fNTried = fDelaunay->GetNTried();
   fMTried = fDelaunay->GetMTried();
   Double_t cp = TMath::Cos(view->GetLongitude()*TMath::Pi()/180.);
   Double_t sp = TMath::Sin(view->GetLongitude()*TMath::Pi()/180.);
   order = new Int_t[fNdt];
   dist  = new Double_t[fNdt];
   Double_t xd,yd;
   Int_t P, N, M;
   Bool_t o = kFALSE;
   for (IT=0; IT<fNdt; IT++) {
      P = fPTried[IT];
      N = fNTried[IT];
      M = fMTried[IT];
      xd = (fXN[P]+fXN[N]+fXN[M])/3;
      yd = (fYN[P]+fYN[N]+fYN[M])/3;
      if ((cp >= 0) && (sp >= 0.)) {
         dist[IT] = -(fXNmax-xd+fYNmax-yd);
      } else if ((cp <= 0) && (sp >= 0.)) {
         dist[IT] = -(fXNmax-xd+yd-fYNmin);
         o = kTRUE;
      } else if ((cp <= 0) && (sp <= 0.)) {
         dist[IT] = -(xd-fXNmin+yd-fYNmin);
      } else {
         dist[IT] = -(xd-fXNmin+fYNmax-yd);
         o = kTRUE;
      }
   }
   TMath::Sort(fNdt, dist, order, o);
   
   // Draw the triangles and markers if requested
   fGraph2D->SetFillColor(fGraph2D->GetFillColor());
   Int_t FS = fGraph2D->GetFillStyle();
   fGraph2D->SetFillStyle(1001);
   fGraph2D->TAttFill::Modify();
   fGraph2D->SetLineColor(fGraph2D->GetLineColor());
   fGraph2D->TAttLine::Modify();
   Int_t LS = fGraph2D->GetLineStyle();
   for (IT=0; IT<fNdt; IT++) {
      T[0] = fPTried[order[IT]];
      T[1] = fNTried[order[IT]];
      T[2] = fMTried[order[IT]];
      for (Int_t t=0; t<3; t++) {
         if(fX[T[t]-1] < fXmin || fX[T[t]-1] > fXmax) goto endloop;
         if(fY[T[t]-1] < fYmin || fY[T[t]-1] > fYmax) goto endloop;
         temp1[0] = fX[T[t]-1];
         temp1[1] = fY[T[t]-1];
         temp1[2] = fZ[T[t]-1];
         temp1[0] = TMath::Max(temp1[0],fXmin);
         temp1[1] = TMath::Max(temp1[1],fYmin);
         temp1[2] = TMath::Max(temp1[2],fZmin);
         temp1[2] = TMath::Min(temp1[2],fZmax);
         if (Hoption.Logx) temp1[0] = TMath::Log10(temp1[0]);
         if (Hoption.Logy) temp1[1] = TMath::Log10(temp1[1]);
         if (Hoption.Logz) temp1[2] = TMath::Log10(temp1[2]);
         view->WCtoNDC(temp1, &temp2[0]);
         x[t] = temp2[0];
         y[t] = temp2[1];
      }
      x[3] = x[0];
      y[3] = y[0];
      if (tri1 || tri2) PaintLevels(T,x,y);
      if (!tri1 && !tri2 && !wire) {
         gPad->PaintFillArea(3,x,y);
         PaintLevels(T,x,y,nblev,glev);
      }
      if (!tri2) gPad->PaintPolyLine(4,x,y);
      if (markers) {
         if (markers0) {
            PaintPolyMarker0(3,x,y);
         } else {
            fGraph2D->SetMarkerStyle(fGraph2D->GetMarkerStyle());
            fGraph2D->SetMarkerSize(fGraph2D->GetMarkerSize());
            fGraph2D->SetMarkerColor(fGraph2D->GetMarkerColor());
            fGraph2D->TAttMarker::Modify();
            gPad->PaintPolyMarker(3,x,y);
         }
      }
endloop:
      continue;
   }
   fGraph2D->SetFillStyle(FS);
   fGraph2D->SetLineStyle(LS);
   fGraph2D->TAttLine::Modify();
   fGraph2D->TAttFill::Modify();
   delete [] order;
   delete [] dist;
   if (glev) delete [] glev;
}
