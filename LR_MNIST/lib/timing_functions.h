// T. Grenier - Clanu 2017-2018 - INSA Lyon - GE

///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef _timing_functions_h
#define _timing_functions_h


#include <ctime>
#include <ratio>
#include <chrono>


std::chrono::steady_clock::time_point tic_time[10];
std::chrono::steady_clock::time_point tac_time[10];

//! Start the chronometer number \param nb.
void tic(unsigned int nb)
{
    tic_time[nb] = std::chrono::steady_clock::now();
}

//! Stop the chronometer number \param nb.
void tac(unsigned int nb)
{
    tac_time[nb] = std::chrono::steady_clock::now();
}

//! Return time in seconds elapsed between last tic(\param nb) and tac(\param nb) calls.
double duration(unsigned int nb)
{
    std::chrono::duration<double> time_span = std::chrono::duration_cast< std::chrono::duration<double> >(tac_time[nb] - tic_time[nb]);
    return time_span.count();
}

//! Start the chronometer.
void tic()
{
    tic(0);
}

//! Stop the chronometer.
void tac()
{
    tac(0);
}

//! Return time in seconds elapsed between last tic() and tac() calls.
double duration()
{
    return duration(0);
}

#endif 
