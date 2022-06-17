/*#include <vector>
#include <T2Dump/DataHandler.h>
#include <exception>

#ifndef _DataIterator_
#define _DataIterator_

class DataIterator
{
private:
  DataHandler* fParent = 0;
  std::vector<T2>::iterator fFirstArrayIterator;
  std::vector<T2>::iterator fSecondArrayIterator;
  
  bool fFirst = true;

public:
  DataIterator() {}
  ~DataIterator() {}

  DataIterator(const DataIterator& d) : 
    fParent(d.fParent),
    fFirstArrayIterator(d.fFirstArrayIterator),
    fSecondArrayIterator(d.fSecondArrayIterator),
    fFirst(d.fFirst)
  {
  }

  DataIterator(DataHandler* base) : 
    fParent(base), 
    fFirstArrayIterator(base->fFirstSecond.end()), 
    fSecondArrayIterator(base->fSecondSecond.end()),
    fFirst(false)
    {}

  void SetParent(DataHandler* base) { fParent = base; }
  
  void LoadNext() 
  {
    if (fFirst)
      return;

    int nInArray = ToInt() - fParent->fFirstSecond.size();
    auto vectors = fParent->ReadNextSecond();

    fFirst = true;
    fFirstArrayIterator = vectors.first->begin();
    fSecondArrayIterator = vectors.second->begin();

    (*this) += nInArray;
  }

  void SetToBegin() 
  { 
    fFirst = true; 
    fFirstArrayIterator = fParent->fFirstSecond.begin(); 
    fSecondArrayIterator = fParent->fSecondSecond.begin();
  }

  int
  ToInt() const
  {
    if (fFirst) 
      return fFirstArrayIterator - fParent->fFirstSecond.begin();
    else
      return fParent->fFirstSecond.size() + fSecondArrayIterator - fParent->fSecondSecond.begin();
  }

  int
  ToIntInSecond() const
  {
    if (fFirst)
      return -1;
    else
      return fSecondArrayIterator - fParent->fSecondSecond.begin(); 
  }

  T2& operator*() 
  { 
    if (fFirst) {
      if (fFirstArrayIterator == fParent->fFirstSecond.end() || 
          !fParent->fFirstSecond.size())
        throw std::out_of_range("trying to access beyond vector limits.");
      return *(fFirstArrayIterator); 
    } else {
      if (fSecondArrayIterator == fParent->fSecondSecond.end() ||
          !fParent->fSecondSecond.size())
        throw std::out_of_range("trying to access beyond vector limits.");
      return *(fSecondArrayIterator);
    }
  }

  T2* operator->()
  {
    if (fFirst) {
      if (fFirstArrayIterator == fParent->fFirstSecond.end())
        throw std::out_of_range("trying to access beyond vector limits.");
      return &(*fFirstArrayIterator);
    } else {
      if (fSecondArrayIterator == fParent->fSecondSecond.end())
        throw std::out_of_range("trying to access beyond vector limits.");
      return &(*fSecondArrayIterator);
    }
  }

  DataIterator& operator++()
  {
    if (fFirst) {
      ++fFirstArrayIterator;
      if (fFirstArrayIterator != fParent->fFirstSecond.end()) {
        return *this;
      } else {
        fFirst = false;
        fSecondArrayIterator = fParent->fSecondSecond.begin();
      }
    } else if (fSecondArrayIterator != fParent->fSecondSecond.end()) {
      ++fSecondArrayIterator;
    }

    return *this;
  }

  DataIterator& operator--()
  {
    if (fFirst) {
      if (fFirstArrayIterator == fParent->fFirstSecond.begin())
        return *this;
      else 
        --fFirstArrayIterator;
    } else {
      if (fSecondArrayIterator == fParent->fSecondSecond.begin()) {
        fFirst = true;
        fFirstArrayIterator = --fParent->fFirstSecond.end();
      } else {
        --fSecondArrayIterator;  
      }      
    }
    return *this;
  }

  DataIterator operator+(int i) 
  {
    auto tmp(*this);
    for (int j = 0; j < abs(i); ++j) {
      if (i > 0)
        ++tmp;
      else
        --tmp;
    }
    return tmp;
  }

  DataIterator operator-(int j)
  {
    return *this + (j*(-1));
  }

  DataIterator& operator+=(int i) 
  {
    for (int j = 0; j < abs(i); ++j) {
      if (i > 0) 
        ++(*this);
      else
        --(*this);
    }

    return *this;
  }

  DataIterator& operator-=(int i)
  {
    return (*this) += (i*(-1));
  }

  DataIterator operator++(int) { auto tmp = *this; ++(*this); return tmp; }
  DataIterator operator--(int) { auto tmp = *this; --(*this); return tmp; }

  bool IsSecond() { return !fFirst; }

  //returns time difference in us
  int 
  GetTimeDifference(DataIterator& b)
  {
    if (fParent != b.fParent) {
      throw std::invalid_argument("Comparing different data sets!");
      return 0;
    }

    if (fFirst == b.fFirst) {
      return (*this)->fTime - b->fTime;
    } else {
      if (fFirst) {
        return (*this)->fTime - 1000000 - b->fTime;
      } else {
        return (*this)->fTime + 1000000 - b->fTime;
      }
    }
  }

  unsigned int
  GetSize() const
  {
    return fParent->fFirstSecond.size() + fParent->fSecondSecond.size();
  }
  
  friend int operator-(const DataIterator& a, const DataIterator& b)
  {
    if (a.fParent != b.fParent) {
      throw std::invalid_argument("Comparing different data sets!");
      return 0;
    }

    return a.ToInt() - b.ToInt();
  }

  friend bool operator==(const DataIterator& a, const DataIterator& b) 
  { 
    if (!(a.fParent == b.fParent && a.fFirst == b.fFirst)) {
      return false;
    } else {
      if (a.fFirst)
        return a.fFirstArrayIterator == b.fFirstArrayIterator;
      else
        return a.fSecondArrayIterator == b.fSecondArrayIterator;
    } 
  }

  friend bool operator!=(const DataIterator& a, const DataIterator& b) 
  {
    if (a.fParent != b.fParent)
      return true;

    if (a.fFirst != b.fFirst) 
      return true;

    if (a.fFirst) {
      return a.fFirstArrayIterator != b.fFirstArrayIterator;
    } else {
      return a.fSecondArrayIterator != b.fSecondArrayIterator; 
    }
  }
};
#endif*/