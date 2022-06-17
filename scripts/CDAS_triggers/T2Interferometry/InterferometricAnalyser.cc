#include <T2Interferometry/InterferometricAnalyser.h>
#include <iostream>
#include <utl/Accumulator.h>
#include <T2Interferometry/Utl.h>

InterferometricAnalyser::InterferometricAnalyser(std::string outbase) : 
  fGalacticDirections("galactic", "galactic", 5),
  fLocalSkyCoordinates("local","local", 5),
  foutBaseName(outbase),
  fOutPut(outbase + ".root")
{
}

InterferometricAnalyser::InterferometricAnalyser(std::string outbase, IDataHandler* h) : 
  fDataHandler(h),
  fGalacticDirections("galactic", "galactic", 5),
  fLocalSkyCoordinates("local", "local", 5),
  foutBaseName(outbase),
  fOutPut(outbase + ".root")
{
}

InterferometricAnalyser::~InterferometricAnalyser() 
{
  fOutPut.Close();
}


//Saves candidate to histograms and tree after applying cuts on size and quality
void
InterferometricAnalyser::SaveEvent(TimeStamp& candidate, 
                                   const std::pair<double, double>& PairThetaPhi, 
                                   const std::vector<double>& raDec)
{
  candidate.fchiSquare = candidate.GetChiSquare();
  // for 30,60 binning
  if (candidate.size() >= fThreshold 
    && candidate.fTimeSpread < 2*(candidate.size() - 3.)
    && candidate.fTimeSpread < 40.) {  
       
    fGalacticDirections.Fill(raDec.front(), raDec.back());
    fLocalSkyCoordinates.Fill(PairThetaPhi.first, PairThetaPhi.second);
    candidate.fAvgDistance = candidate.GetAvgDistance(fDataHandler->fPositions);

    fOutPut << candidate;
  }
}

void
InterferometricAnalyser::FindEvents(uint bin)
{
  //these things do not change during this part of the analysis
  TimeStamp candidate;
  candidate.fGPSSecond = fDataHandler->fGPSSecond;
  candidate.fDirectionBin = bin;

  //conversion of current direction to Equitorial coordinates (ra dec)
  const auto PairThetaPhi = fDataHandler->GetBinCenter(bin);
  std::vector<double> input;
  input.push_back(candidate.fGPSSecond);
  input.push_back(PairThetaPhi.first);   //was acos
  input.push_back(PairThetaPhi.second);
  const auto raDec = ConvertToRaDec(input);

  //const auto galactic = GetGalacticCoordinates(input);

  if (fDataHandler->fGPSSecond == fPrintSecond) {
    std::cerr << std::endl << bin << " ";
  }

  utl::Accumulator::DataAverage tRef;
  //scan through sorted t0's and find peaks
  for (const auto& t2 : fDataHandler->fProjectedMicroSeconds.at(bin)) {
    double t = t2.fTime;
    const double sigmaTref = fDataHandler->fPositions.GetSigmaTref(bin, t2.fId);
    
    if (fDataHandler->fGPSSecond == fPrintSecond) {
      std::cerr << t2.fTime << " " << sigmaTref << " "; 
    }

    if (!candidate.fIds.size()) {
      candidate.fIds.push_back(t2.fId);
      candidate.fReconstructedTrefs.push_back(std::make_pair(t, sigmaTref));
      tRef(t, sigmaTref);

      if (fDataHandler->fGPSSecond == fPrintSecond) {
        std::cerr << sigmaTref << " ";
      }
    } else {
      //calculation of uncorrelated parts of the variance
      //const auto basis = Get2DBasis(fDataHandler->fPositions.GetPosition(t2.fId),
      //                                fDataHandler->fPositions.GetPosition(candidate.fIds.back()));
      const double varTrefUncorrelated = fDataHandler->fPositions.GetVarAX(bin, 
                                         fDataHandler->fPositions.GetDeltaX(t2.fId,
                                                              candidate.fIds.back())) + 1;    
      //also accounts for rounding of time

      if (fDataHandler->fGPSSecond == fPrintSecond) {
        std::cerr << sqrt(varTrefUncorrelated) << " ";
      }

      //take 2 sigma as initial cut, is refined in RemoveAccidentals()
      if (pow(candidate.fReconstructedTrefs.back().first - t, 2) 
          <= 4*varTrefUncorrelated) {   
        candidate.fIds.push_back(t2.fId);
        candidate.fReconstructedTrefs.push_back(std::make_pair(t, varTrefUncorrelated));
        tRef(t, varTrefUncorrelated);
      } else {
        candidate.fMicroSecond = tRef.GetMean();
        auto remainingT2s = candidate.RemoveAccidentals();

        while (remainingT2s.size() >= fThreshold) {
          auto removedT2s = remainingT2s.RemoveAccidentals();
          SaveEvent(remainingT2s, PairThetaPhi, raDec);
          remainingT2s = removedT2s;
        }
        SaveEvent(candidate, PairThetaPhi, raDec);
        
        //reset the current candidate, to include only the current t2
        candidate.reset();
        candidate.fGPSSecond = fDataHandler->fGPSSecond;
        candidate.fDirectionBin = bin;
        candidate.fIds.push_back(t2.fId);
        candidate.fReconstructedTrefs.push_back(std::make_pair(t, sigmaTref));

        tRef.Clear();
        tRef(t, sigmaTref);
      }
    }
  }
}

void
InterferometricAnalyser::Analyse(uint firstGPSs, uint lastGPSs)
{
  std::cout << "Processing GPS second: " << fDataHandler->fGPSSecond << std::endl;

  while (fDataHandler->fGPSSecond < firstGPSs) {
    fDataHandler->ReadNextSecond();
    if (!(fDataHandler->fGPSSecond % 10))
      std::cout << "Processing GPS second: " << fDataHandler->fGPSSecond << std::endl;
  }

  while (!fDataHandler->fEoF) {
    for (uint i = 0; i < fDataHandler->GetNBins(); ++i) {
      FindEvents(i);
    }

    fDataHandler->ReadNextSecond();

    if (!(fDataHandler->fGPSSecond % 1))
      std::cout << "Processing GPS second: " << fDataHandler->fGPSSecond << std::endl;  
    if (fDataHandler->fGPSSecond > lastGPSs)
      break;
  }

  fOutPut.Write(fGalacticDirections);
  fOutPut.Write(fLocalSkyCoordinates);
}
