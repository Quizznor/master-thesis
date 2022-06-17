#include <TVector3.h>
#include <TMath.h>
#include <TCanvas.h>
#include <TGraph.h>
#include <iostream>
#include <TSystem.h>
#include <TROOT.h>

static TVector3 x[3];

void 
Init(const int& one = 0,const int& two = 0)
{
  x[0].SetXYZ(0,0,0);
  x[1].SetXYZ(0., (one + 1)*1500., 0.);
  x[2].SetXYZ((two + 1)*750, sqrt(3)/2.*(two + 1)*1500, 0);
}

template <typename T>
T sqr(const T& t)
{
  return t*t;
}

double
GetAlpha(const double& tau1, const double& tau2, const double& delta1, const double& delta2, const double& ij)
{
  return (tau1 - tau2*ij + delta1 - ij*delta2)/(1- ij);
}

double
GetBeta(const double& tau1, const double& tau2, const double& delta1, const double& delta2, const double& ij)
{
  return GetAlpha(tau2, tau1, delta2, delta1, ij);
}

double
GetGamma(const double& alpha, const double& beta, const double& ij)
{
  double tmp = 1 - sqr(alpha) - sqr(beta) - 2*alpha*beta*ij;
  if (tmp < 0) {
    //std::cerr << "warning: subluminal event!" << std::endl;
    return -1;
  }
  if (ij > 0.99) {
    //std::cerr << "warning: stations aligned!" << std::endl;
    return -1;
  }

  return sqrt(tmp/(1 - sqr(ij)));
}

TVector3
GetAxis(const double& alpha, const double& beta, const TVector3& i, const TVector3& j)
{ 
  const double gamma = GetGamma(alpha, beta, i*j);
  if (gamma == -1)
    return TVector3();
  return alpha*i + beta*j + gamma*i.Cross(j);
}

void
ScanDirections(const double& t1, const double& t2)
{	
  TGraph ScatterPlot;
  THealPixD test("test","test",4);
  test.SetDegree();

  const TVector3 x21 = x[1] - x[0];
  const TVector3 x31 = x[2] - x[0];

  const TVector3 iVec = x21*(1./x21.Mag());
  const TVector3 jVec = x31*(1./x31.Mag());

  const double tau1 = 300./x21.Mag()*t1;
  const double tau2 = 300./x31.Mag()*t2;

  for (double  delt1 = -0.5; delt1 < 0.5; delt1 += 0.01) {
    for (double delt2 = -0.5; delt2 < 0.5; delt2 += 0.01) {
      const double del1 = 300./x21.Mag() * delt1;
      const double del2 = 300./x31.Mag() * delt2;

      const TVector3 iVec = x21*(1./x21.Mag());
      const TVector3 jVec = x31*(1./x31.Mag());

      const TVector3 axis = GetAxis(GetAlpha(tau1, tau2, del1, del2, iVec*jVec), GetBeta(tau1, tau2, del1, del2, iVec*jVec), iVec, jVec);
      ScatterPlot.SetPoint(ScatterPlot.GetN(), 180. - axis.Theta()*180./TMath::Pi(), axis.Phi()*180./TMath::Pi());
      
      if (axis.Mag() > 0)
        test.Fill(180. - axis.Theta()*180./TMath::Pi(), axis.Phi()*180./TMath::Pi());
      if (!(ScatterPlot.GetN() % 25))
        cout << delt1 << " , " << delt2 << endl;
      if (delt1 > -0.005 && delt1 < 0.005 && delt2 > -0.005 && delt2 < 0.005)
        cout << "initial: " << 180. - axis.Theta()*180./TMath::Pi() << " , " << axis.Phi()*180./TMath::Pi() << endl; 
    }
  }

  //TCanvas* c = new TCanvas();
  ScatterPlot.SetMarkerStyle(5);
  ScatterPlot.SetTitle(";#Theta/#circ;#Phi/#circ");
  //c->cd();
  //ScatterPlot.DrawClone("AP");

  TCanvas* c= new TCanvas();
  c->cd();
  test.DrawCopy("colzHAMMERCELESTIAL");
}



















