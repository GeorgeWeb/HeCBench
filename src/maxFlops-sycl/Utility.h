#ifndef UTILITY_H
#define UTILITY_H

#ifdef _WIN32

// On Windows, srand48 and drand48 don't exist.
// Create convenience routines that use srand/rand
// and let developers continue to use the -48 versions.

inline void srand48(unsigned int seed)
{
    srand(seed);
}

inline double drand48()
{
    return double(rand()) / RAND_MAX;
}

#endif // _WIN32

#endif // UTILITY_H
