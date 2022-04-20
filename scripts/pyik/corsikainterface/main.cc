#include <CorsikaShower.h>
#include <CorsikaShowerFile.h>
#include <string>
#include <iostream>

using namespace io;
using namespace std;

int main(int argc, char* argv[])
{
  string filename;
  if (argc == 1)
    filename = "/home/jgonzalez/corsika/maximo/DAT002699.part";
  else
    filename = argv[1];
  if(CorsikaShowerFile::IsValid(filename)) {
    CorsikaShowerFile file(filename);

    file.Read();
    cout << "Primary: " << file.GetCurrentShower().GetPrimary() << endl;
    cout << "Energy: " << file.GetCurrentShower().GetEnergy() << endl;
    cout << "MuonNumber: " << file.GetCurrentShower().GetMuonNumber() << endl;
    cout << "Zenith: " << file.GetCurrentShower().GetZenith() << endl;
    cout << "Azimuth: " << file.GetCurrentShower().GetAzimuth() << endl;
    cout << "MinRadiusCut: " << file.GetCurrentShower().GetMinRadiusCut() << endl;
    cout << "ShowerNumber: " << file.GetCurrentShower().GetShowerNumber() << endl;
    cout << "EMEnergyCutoff: " << file.GetCurrentShower().GetEMEnergyCutoff() << endl;
    cout << "MuonEnergyCutoff: " << file.GetCurrentShower().GetMuonEnergyCutoff() << endl;

    ShowerParticleList<CorsikaShowerFileParticleIterator> particles = file.GetCurrentShower().GetParticles();

    int count = 0;
    for (ShowerParticleIterator it = particles.begin(); it != particles.end(); ++it) {
      if (count < 2) {
        cout << it->fWeight << endl;
      }
      ++count;
    }
    cout << count << " particles" << endl;
  }
  else
    cout << "File " << filename << " is not a Corsika file" << endl;
  return 0;
}
