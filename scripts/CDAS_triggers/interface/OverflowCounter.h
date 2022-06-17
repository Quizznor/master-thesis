#ifndef _Overflow_
#define _Overflow_


/**
 * @brief Counter that has a limited range and can be in/de-cremented over that (->starts again at zero)
 * @details intended for the use in a circular array, this counter provides increment/decrement +=/-= 
 *          while staying inside [0, size)
 *          feature: if incremented above size, it starts again at zero (similar for decrement)
 */
struct OverflowCounter{
  std::size_t value = 0;
  std::size_t size = 1;

  OverflowCounter() {}
  OverflowCounter(std::size_t Size) : size(Size) {}
  ~OverflowCounter(){}

  OverflowCounter& operator++() { value = ++value % size; return *this; }
  OverflowCounter operator++(int) { OverflowCounter tmp = *this; ++(*this); return tmp; }
  
  OverflowCounter& operator--() { 
    if ( value > 0) {
      value = --value;
    } else {
      value = size - 1;
    }
    return *this;
  }
  OverflowCounter operator--(int) { OverflowCounter tmp = *this; --(*this); return tmp; }

  OverflowCounter& operator+=(std::size_t i) { value = (value + i) % size; return *this; }
  
  OverflowCounter& operator+=(const OverflowCounter& rhs) {
    value = (value + rhs.value) % size;
    return *this;
  }

  OverflowCounter& operator-=(std::size_t i){ 
    if (value >= i) {
      value -= i;
    } else {
      value = size - (i - value );
    }
    return *this;
  }

  OverflowCounter& operator-=(const OverflowCounter& rhs){
    return (*this) -= rhs.value;
  }

  friend OverflowCounter& operator+(OverflowCounter lhs, uint i) { return OverflowCounter(lhs) += i; }
  friend OverflowCounter& operator-(OverflowCounter lhs, uint i) { return OverflowCounter(lhs) -= i; }

  friend int operator-(const OverflowCounter& lhs, const OverflowCounter& rhs) { 
    if (lhs.value >= rhs.value) 
      return lhs.value - int(rhs.value);
    else
      return rhs.size - int(rhs.value) + lhs.value; 
  }

  friend bool operator==(const OverflowCounter& lhs, const OverflowCounter& rhs) { return (lhs.size == rhs.size && lhs.value == rhs.value); }
  friend bool operator!=(const OverflowCounter& lhs, const OverflowCounter& rhs) { return !(rhs==lhs); }

  operator int() const { return int(value); }
  operator std::size_t() const { return size; }
  operator uint() const { return uint(value); }
};
#endif