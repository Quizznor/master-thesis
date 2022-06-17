#include <interface/OverflowCounter.h>

#ifndef _iterator_
#define _iterator_

/**
 * @brief iterator class for circ_array
 * @details (almost?) random access iterator for circ_array (c.f. circ_array.h) always a forward iterator, although rbegin and rend exist
 * taking care of direction (increment and decrement) is up to the user 
 * Will not be increased after end and not decreased after begin (member max_in/de-crease take care of that)
 * 
 * @tparam std::N size of the circ_array 
 * @tparam T type of data
 */

template <std::size_t N, typename T>
class circ_array;

template < std::size_t N, typename T >
class circ_iterator{
  private:
    circ_array<N, T>* c = nullptr;
    OverflowCounter element;
    std::size_t max_increase = 0;
    std::size_t max_decrease = 0;
    
  public:
    circ_iterator() : element(N + 1) {}
    ~circ_iterator() {}

    T& operator*() { return c->fData.at(int(element)); }
    T* operator->() { return &(c->fData.at(int(element))); }

    circ_iterator& operator++() { 
      if (max_increase > 0) {
        ++element;
        --max_increase;
        ++max_decrease;
      }
      return *this;
    }
    circ_iterator& operator--() { 
      if (max_decrease > 0) {
        --element; 
        --max_decrease;
        ++max_increase;
      }
      return *this; 
    }

    circ_iterator& operator+=(std::size_t i) {
      if ( i > max_increase) {
        element += max_increase;
        max_decrease += max_increase;
        max_increase = 0;
      } else {
        element += i;
        max_increase -= i;
        max_decrease += i;
      }      
      return *this; 
    }
    circ_iterator& operator-=(std::size_t i) { 
      if ( i > max_decrease) {
        element -= max_decrease;
        max_increase += max_decrease;
        max_decrease = 0;
      } else {
        element -= i;
        max_increase += i;
        max_decrease -= i;
      } 
      return *this; 
    }

    void SetPtr(circ_array<N, T>* initPtr) { c = initPtr; }
    void SetElement(OverflowCounter el) { element = el; }
    void SetMaxSteps(std::size_t inc, std::size_t dec) { max_decrease = dec; max_increase = inc; }

    friend bool operator==(const circ_iterator& a, const circ_iterator& b) { return (a.c == b.c && a.element.value == b.element.value);}
    friend bool operator!=(const circ_iterator& a, const circ_iterator& b) { return !(a == b);}
};

#endif