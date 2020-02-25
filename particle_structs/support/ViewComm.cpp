#include "ViewComm.h"
#include <unordered_map>
#include <mpi.h>
#include <functional>
namespace pumipic {

  Irecv_Map lambda_map;

  Irecv_Map& get_map() {return lambda_map;}
}
