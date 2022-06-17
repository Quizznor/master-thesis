#include <vector>
#include <stdexcept>
#include <interface/OverflowCounter.h>
#include <interface/circ_iterator.h>

typedef unsigned int uint;

#ifndef _circ_array_
#define _circ_array_

/**
 * @brief buffer like data structure with fixed size and iteratable
 * @details based on an array (fixed size) this data structure provides:
 *              - push_back() in O(1)
 *              - pop_back() in O(1)
 *              - pop_front() in O(1)
 *              - front()/back() in O(1)
 *              - push_front() in O(1) would be possible (not implemented)
 *              - random_access with operator[] or at in O(1)
 *              - efficient (?) iteration over content with iterators (circ_iterator class)
 * 
 * @tparam  std::N size of buffer
 * @tparam T  data type of content
 */
template < std::size_t N, typename T >
class circ_array
{
private:
  std::vector<T> fData;

  OverflowCounter fbegin;
  OverflowCounter fend;

  const std::size_t fsize = N;
public:
  circ_array() : fData(N + 1), fbegin(N + 1), fend(N + 1) {}
  ~circ_array() {}
  
  void 
  push_back(T data) 
  { 
    fData[int(fend)] = data;
    ++fend;

    if ( int(fend) == int(fbegin)) 
      ++fbegin;
  }

  T 
  pop_back() 
  {
    if ( int(fbegin) != int(fend) ) {
      return fData[int(--fend)];
    } else {
      throw std::out_of_range("Accessing empty container");
    }
    return 0;
  }

  T 
  pop_front() 
  {
    if ( int(fbegin) != int(fend) ){
      return fData[int(fbegin++)];
    } else {
      throw std::out_of_range("Accessing empty container");
    }
    return 0;
  }

  T& 
  front() 
  {
    if ( int(fbegin) != int(fend) )
      return fData[int(fbegin)];
    else 
      throw std::out_of_range("Accessing empty container");
    return fData[int(fend)];
  }

  T& 
  back() 
  {
    if ( int(fbegin) != int(fend)) {
      OverflowCounter tmp = fend;
      --tmp;
      return fData[int(tmp)];
    }
    else 
      throw std::out_of_range("Accessing empty container");
    return fData[int(fend)];
  }

  T& 
  at(std::size_t index)
  {
    if (index < fend - fbegin)
      return fData.at(fbegin + index);
    else
      throw std::out_of_range("Index out of Range");
    return fData.at(fbegin + index);
  }

  T& 
  operator[](std::size_t index) 
  {
    OverflowCounter tmp = fbegin;
    tmp += index;
    return fData[int(tmp)];
  }

  circ_iterator<N, T> 
  begin() 
  {
    auto it = circ_iterator<N, T>();
    it.SetPtr(this);
    it.SetElement(fbegin);
    it.SetMaxSteps(this->GetN(), 0);  //the end element is a valid iterator, so one more than elements
    return it;
  }

  circ_iterator<N, T> 
  rbegin()
  {
    auto it = circ_iterator<N, T>();
    it.SetPtr(this);
    OverflowCounter tmp = fend;
    it.SetElement(--tmp);
    it.SetMaxSteps(0, this->GetN());  //the end element is a valid iterator, so one more than elements
    return it;
  }

  circ_iterator<N, T> 
  end() 
  {
    auto it = circ_iterator<N, T>();
    it.SetPtr(this);
    it.SetElement(fend);
    it.SetMaxSteps(0, this->GetN());
    return it;
  }

  circ_iterator<N, T> 
  rend() 
  {
    auto it = circ_iterator<N, T>();
    it.SetPtr(this);
    OverflowCounter tmp = fbegin;
    it.SetElement(--tmp);
    it.SetMaxSteps(this->GetN(), 0);
    return it;
  }

  std::size_t size() { return fData.size() - 1; }
  std::size_t GetN() { return fend - fbegin ; }
  bool empty() { return fbegin == fend; }

  template< std::size_t n, typename t>
  friend class circ_iterator;
};

#endif
