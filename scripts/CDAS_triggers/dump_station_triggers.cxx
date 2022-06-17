#include <T2Dump/Utl.h>

#include <t2/T2DumpFile.h>
#include <t2/RawT2DumpFile.h>
#include <t2/SimpleDataStorage.h>
#include <t2/StationInfo.h>

#include <io/zfstream.h>

#include <utl/LineStringTo.h>

#include <boost/program_options.hpp>
#include <algorithm>


using namespace io;
using namespace utl;
using namespace t2;
namespace bpo = boost::program_options;
using namespace std;


typedef long datatype;
typedef StationInfo<datatype> SI;

int
main(int argc, char* argv[])
{
  vector<string> input;
  string outputbase;

  vector<int> ids;

  bpo::options_description options("Usage");
  options.add_options()
    ("help,h",
     "Output this help.")
    ("input,i",
     bpo::value<vector<string>>(&input)->required(),
     "Input filenames.")
    ("output,o",
     bpo::value<string>(&outputbase)->required(),
     "outputbase name.")
    ("id",
      bpo::value<vector<int>>(&ids),
      "ids of the station to dump")
    ("pandas",
     "don't use TTree style format description but pandas compatible one")
    ;
  bpo::positional_options_description positional;
  positional.add("input", -1);
  bpo::variables_map vm;
  try {
    bpo::store(
      bpo::command_line_parser(argc, argv).options(options).positional(positional).run(),
      vm
    );
    bpo::notify(vm);
  } catch (bpo::error& err) {
    cerr << "Command line error : " << err.what() << '\n'
         << options << endl;
    exit(EXIT_FAILURE);
  }

  if (vm.count("help") || input.empty()) {
    cerr << options << endl;
    return EXIT_SUCCESS;
  }

  std::cout << "got " << ids.size() << " station ids" << std::endl;

  string outputName = outputbase + ".dat";
  ofstream outstream(outputName);
  if (vm.count("pandas"))
    outstream << "GPSSecond Id Time Type\n";
  else
    outstream << "fGPSSecond/I:fId:fTime:fType\n";

  string outputNameNumber = outputbase + "_ntriggers.dat";
  ofstream outstream2(outputNameNumber);
  outstream2 << "fGPSSecond/I:fnT2:fnToT:fnTh\n";

  SimpleDataStorage t2data;

  for (const auto& name : input) {
    RawT2DumpFile file(name);

    while (file.ReadNextSecond(t2data)) {
      t2data.Sort();

      int nT2 = 0;
      int nToT = 0;
      int nTh = 0;
      for (const auto& t2 : t2data.fData) {
        if (std::find(ids.begin(), ids.end(), t2.fId) == ids.end())
          continue;

        ++nT2;
        if (t2.fTriggers < 7)
          ++nTh;
        else if (t2.fTriggers > 7)
          ++nToT;

        outstream << t2data.fGPSSecond << ' '
                  << t2.fId << ' '
                  << t2.fTime << ' '
                  << t2.fTriggers << '\n';
      }

      outstream2 << t2data.fGPSSecond << ' '
                 << nT2 << ' '
                 << nToT << ' '
                 << nTh << '\n';
    }
  }


  return EXIT_SUCCESS;
}
