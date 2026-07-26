#include <cstddef>
#pragma once
template <class T>
struct NetNode {
   T z = 0;
   size_t node_id = 0;
   T w = 0;
   double potential = 0;

   NetNode() = default;
   NetNode(T zc, size_t id) : z(zc), node_id(id) {
     w = 0.0;
     potential = 0.0;
   }

   NetNode(T zc, size_t id, T wv) : z(zc), node_id(id), w(wv) {
    potential = 0.0;
   }
};