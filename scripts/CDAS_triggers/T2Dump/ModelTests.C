#include <TMinuit.h>
#include <TVector3.h>
#include <fstream>
#include <iostream>
#include <TGraph2D.h>
#include <TF2.h>
#include <TF1.h>
#include <TString.h>
#include <TCanvas.h>
#include <algorithm>

static TVector3 positions[3];
static int Times[3];
const double sigmaVOverC = 0.005;

struct POS
{
  TVector3 fPosition[2000];
};

static POS pos;

template <typename T>
T sqr(const T& t)
{ 
  return t*t;
}


//should calculate the binned form of a 2d-gaus with .5 correlation, as this is what the two delta t are supposed to look like
double
GetBinIntegralGaus(const double& deltaTHat1, const double& deltaTHat2, const double& delt1, const double delt2)
{
  TF2 gaus("g","1./(sqrt(3)*TMath::Pi())*exp(-2/3.*(pow(x - [0],2) - (x - [0])*(y - [1]) + pow(y - [1],2)))", -100, 100, -100, 100);

  gaus.SetParameter(0, deltaTHat1);
  gaus.SetParameter(1, deltaTHat2);

  return gaus.Integral(floor(delt1), floor(delt1) + 1, floor(delt2), floor(delt2) + 1);
}

double
GetDeltaTHat(const double& theta, const double& phi, const TVector3& deltaX, const double& vOverC)
{
  TVector3 axis;
  axis.SetXYZ(1., 0., 0.);
  axis.SetTheta(theta);
  axis.SetPhi(phi);

  //cout << "deltaX: " << deltaX.x() << " , " << deltaX.y() << endl;

  return -1./(300.*vOverC)*axis*deltaX;
}
/*
  parameter:
   [2] ->Speed of front in units of c (should be 1)
   [0] theta (in rad)
   [1] phi (in rad)
*/

void
fcnWithV(int &npar, double *gin, double &f, double *par, int iflag)
{
  if (npar < 3)
    return;

  double tmp = sqr(1 - par[2])/(2.*sqr(sigmaVOverC));

  double delT12 = Times[1] - Times[0];
  double delT13 = Times[2] - Times[0];

  const TVector3 deltaX12(positions[1] - positions[0]);
  const TVector3 deltaX13(positions[2] - positions[0]);

  double binIntegral = GetBinIntegralGaus(GetDeltaTHat(par[0], par[1], deltaX12, par[2]), GetDeltaTHat(par[0], par[1], deltaX13, par[2]), delT12, delT13);

  if (binIntegral && fabs(binIntegral) < 1e-200) {
    tmp += 500 + log(-log(fabs(binIntegral)) + 1);
    cout << log(-log(fabs(binIntegral)) + 1) << " , " << binIntegral << " , " << -log(fabs(binIntegral)) << endl;
  } else if (!binIntegral) {
    tmp += 1000;
  } else {
    tmp -= log(binIntegral);
  }
  
  f = tmp;
}

//for the fcn Gaus... not using theta and phi
//v in units of c
double
GetTHatZero(const double& u, const double& v, const double& t0, const TVector3& deltaX, const double& speed = 1)
{
  TVector3 axis;
  axis.SetXYZ(u, v, sqrt(1 - sqr(u) - sqr(v)));

  return -axis*deltaX/(300.*speed) + t0;
}

/*
  Replace box with Gaus for numerics, (later maybe use MC in 1 sigma interval to get one/sample of the 'equal' solutions)
    also fit t_0

  Interface:
    par[0] = t_0
    par[1] = u
    par[2] = v
*/
void
fcnGaus(int &npar, double *gin, double &f, double *par, int iflag)
{
  if (npar < 3)
    return;

  const TVector3 deltaX12(positions[1] - positions[0]);
  const TVector3 deltaX13(positions[2] - positions[0]);
  const TVector3 deltaX11(0., 0., 0.);
  
  f = 0;
  if (npar > 3) {
    f += sqr(par[3] - 1)/0.0001;
    f += sqr(Times[0] - GetTHatZero(par[1], par[2], par[0], deltaX11, par[3]));
    f += sqr(Times[1] - GetTHatZero(par[1], par[2], par[0], deltaX12, par[3])); 
    f += sqr(Times[2] - GetTHatZero(par[1], par[2], par[0], deltaX13, par[3]));
  } else {
    f += sqr(Times[0] - GetTHatZero(par[1], par[2], par[0], deltaX11, 1.));
    f += sqr(Times[1] - GetTHatZero(par[1], par[2], par[0], deltaX12, 1.)); 
    f += sqr(Times[2] - GetTHatZero(par[1], par[2], par[0], deltaX13, 1.));
  }
    
}

/*
  Change parameter to phi and theta
   par[0] is t0
   par[1] is theta
   par[2] is phi
   par[3] is speed of front in units of c (may be omitted or not)
*/
void
fcnGausThetaPhi(int &npar, double *gin, double &f, double *par, int iflag)
{
  if (npar < 3)
    return;

  const TVector3 deltaX12(positions[1] - positions[0]);
  const TVector3 deltaX13(positions[2] - positions[0]);
  const TVector3 deltaX11(0., 0., 0.);
  
  const double u = sin(par[1])*cos(par[2]);
  const double v = sin(par[1])*sin(par[2]);

  f = 0;

  if (npar == 4) {
    f += sqr(par[3] - 1)/0.0001;

    f += sqr(Times[0] - GetTHatZero(u, v, par[0], deltaX11, par[3]));
    f += sqr(Times[1] - GetTHatZero(u, v, par[0], deltaX12, par[3])); 
    f += sqr(Times[2] - GetTHatZero(u, v, par[0], deltaX13, par[3]));
  } else {
    f += sqr(Times[0] - GetTHatZero(u, v, par[0], deltaX11, 1));
    f += sqr(Times[1] - GetTHatZero(u, v, par[0], deltaX12, 1)); 
    f += sqr(Times[2] - GetTHatZero(u, v, par[0], deltaX13, 1));
  }
    
}

double
GetTHat(const double& theta, const double& phi, const double& delt, const int& i)
{
  TVector3 axis;
  axis.SetXYZ(1., 0., 0.);
  axis.SetTheta(theta);
  axis.SetPhi(phi);

  return -1./300.*axis*(positions[i - 1] - positions[0]) + Times[0] + delt;
}

void
fcn(int &npar, double *gin, double &f, double *par, int iflag)
{
  if (npar < 3)
    return;

  double tmp = 0;

  tmp += sqr(par[2]);

  tmp += sqr(Times[1] - (GetTHat(par[0], par[1], par[2], 2)));
  tmp += sqr(Times[2] - (GetTHat(par[0], par[1], par[2], 3)));

  f = tmp;
}

void
ReadPositions()
{
  string positionFile = "/home/schimassek/SVN/T2Scalers/ms/src/Data/SdPositions.txt";

  ifstream inPositions(positionFile.c_str());
  double x = 0;
  double y = 0;
  double z = 0;
  int id = 0;

  double tmp = 0;

  while (inPositions >> id >> y >> x >> z >> tmp >> tmp >> tmp >> tmp >> tmp >> tmp) {
    pos.fPosition[id - 2].SetX(x);
    pos.fPosition[id - 2].SetY(y);
    pos.fPosition[id - 2].SetZ(z);
  }
}

void
SetPosition(const int& id, const int& number)
{
  positions[number - 1] = pos.fPosition[id - 2];
}

void
SetTime(const int& muSec, const int& number)
{
  Times[number - 1] = muSec;
}

void
Init()
{
  int ids[3] = { 452, 901, 1668};
  for (int i = 0; i < 3; ++i) {
    SetTime(0, i + 1);
    SetPosition(ids[i], i + 1);
  }
}

bool
GetStartValues(double& uStart, double& vStart, double& speedStart)
{
  TVector3 axis;

  const TVector3 x12(positions[1] - positions[0]);
  const TVector3 x13(positions[2] - positions[0]);

  const TVector3 iVec(x12*(1./x12.Mag()));
  const TVector3 jVec(x13*(1./x13.Mag()));

  double ij = iVec*jVec;
  if (ij > 0.99) { //prevent bad starting values
    ij = 0.99;
  }

  speedStart = 1;

  double maxDelx = std::max(x12.Mag(), x13.Mag());
  double delTmax = std::max(abs(Times[1] - Times[0]), abs(Times[2] - Times[0]));

  cout << "maxDelx / c (mu s): " << maxDelx/300. << " , Max delta T: " << delTmax << endl;

  if (delTmax > maxDelx/300.) {
    speedStart = maxDelx/(300.*delTmax);
    cout << "slower than speed of light " << endl;
  }

  const double tau12 = 300.*speedStart*(Times[0] - Times[1])/x12.Mag();
  const double tau13 = 300.*speedStart*(Times[0] - Times[2])/x13.Mag();

  double D = 1 - sqr(ij);

  double alpha = (tau12 - tau13*ij)/D;
  double beta = (tau13 - tau12*ij)/D;

  double gammaSqr = (1 - (alpha*iVec + beta*jVec).Mag())/sqr((iVec.Cross(jVec)).Mag());

  TVector3 initAxis;
  initAxis.SetXYZ(0, 0, 0);
  initAxis += alpha*iVec;
  initAxis += beta*jVec;


  if (gammaSqr > 1e-10) { //add sign choice, so that initAxis.z > 0
    if (iVec.Cross(jVec).z() > 0) {
      initAxis += sqrt(gammaSqr)*(iVec.Cross(jVec));
    } else {
      initAxis -= sqrt(gammaSqr)*(iVec.Cross(jVec));
    }
  } else if (gammaSqr < 0) {
    cout << "warning" << endl;

    uStart = 3.1415/2.;
    vStart = 3.1415/180.*330.;

    return false;
  }
  
  if (initAxis.z() < 0)
    initAxis *= -1;

  cout << "Start Axis: " << initAxis.x() << " , " << initAxis.y() << " , " << initAxis.z() << endl;

  uStart = initAxis.x();
  vStart = initAxis.y();

  return true;
}

void
Minimise()
{
  double uStart = 0;
  double vStart = 0;
  double speedStart = 1;
  
  bool vertical = GetStartValues(uStart, vStart, speedStart);


  if (vertical) {
    TMinuit* min = new TMinuit(3);
    min->SetFCN(fcnGaus);

    min->DefineParameter(0, "t0", Times[0], 0.5, -100 + Times[0], Times[0] + 100);
    min->DefineParameter(1, "u", uStart, 0.1, -1, 1);
    min->DefineParameter(2, "v", vStart, 0.001, -1, 1);
    //min->DefineParameter(3, "speed", speedStart, 0.001, 0, 2);
    //min->FixParameter(3);

    min->Migrad();

    double fittedValues[3];
    double fittedErr[3];

    for (int i = 0; i < 3 ; ++i) {
      min->GetParameter(i, fittedValues[i], fittedErr[i]);
    }

    TVector3 recoAxis;
    recoAxis.SetXYZ(fittedValues[1], fittedValues[2], sqrt(1 - sqr(fittedValues[1]) - sqr(fittedValues[2])));

    cout << "Direction: " << recoAxis.Theta()*180./3.1415 << " , " << recoAxis.Phi()*180./3.1415 << endl;
  } else {
    TMinuit* min = new TMinuit(3);

    min->SetFCN(fcnGausThetaPhi);

    min->DefineParameter(0, "t0", Times[0], 0.5, -100 + Times[0], Times[0] + 100);
    min->DefineParameter(1, "theta", uStart, 0.001, -10, 10);
    min->DefineParameter(2, "phi", vStart, 0.5, -10, 10);
    //min->DefineParameter(3, "speed", speedStart, 0.001, 0, 2);

    //min->FixParameter(3);

    min->Migrad();

    double fittedValues[3];
    double fittedErr[3];

    for (int i = 0; i < 3 ; ++i) {
      min->GetParameter(i, fittedValues[i], fittedErr[i]);
    }

    cout << "Direction: " << fittedValues[1]*180/3.1415 << " , " << fittedValues[2]*180./3.1415 << endl;
  }

    
}

void
ScanLikelihood(double speed = 1, double t0 = 0)
{
  TGraph2D graph;

  double gin = 0;
  double logL = 0;

  double par[4];
  par[0] = Times[0] + t0;
  par[1] = -1;
  par[2] = 0;
  par[3] = speed;

  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < 200; ++j) {
      double theta = 3.1415/150.*(i+1);
      double phi = 2*3.1415/180.*(j+1);
      int n = 4;

      TVector3 axis;
      axis.SetXYZ(1, 0 , 0);
      axis.SetTheta(theta);
      axis.SetPhi(phi);

      par[1] = axis.x();
      par[2] = axis.y();

      fcnGaus(n, &gin, logL, par, 0);

      graph.SetPoint(graph.GetN(), theta*180./3.1415, phi*180./3.1415, logL);
    }
  }

  graph.SetMarkerStyle(20);
  graph.DrawClone("Pcol");
}

void
ScanLikelihoodWithV(bool print = true)
{
  TGraph2D graph[4];

  double gin = 0;
  double logL = 0;

  double par[3];

  par[2] = 1;
  par[1] = 0;
  par[0] = 0;

  for (int k = 0; k < 4; ++k) {
    for (int i = 0; i < 100; ++i) {
      for (int j = 0; j < 200; ++j) {
        par[0] = 3.1415/200*(i+1);
        par[1] = 2*3.1415/200*(j+1);
        int n = 3;

        fcnWithV(n, &gin, logL, par, 0);

        if (logL < 1000)
          graph[k].SetPoint(graph[k].GetN(), par[0]*180/3.1415, par[1]*180./3.1415, logL);
      }
    }
    par[2] -= 0.01;
    graph[k].SetMarkerStyle(20);
    TString tmp = "";
    tmp += k;
    tmp += "; #theta/#circ; #phi/#circ";
    graph[k].SetTitle(tmp.Data());
  }
  
  if (print) {
    TCanvas* c = new TCanvas();
    c->Divide(2,2);
    for (int i = 0; i < 4; ++i) {
      c->cd(i + 1);
      graph[i].DrawClone("Pcolz");  
    }
  } else {
    TCanvas* d = new TCanvas();
    d->cd();
    graph[0].DrawClone("colz");
  }
}
