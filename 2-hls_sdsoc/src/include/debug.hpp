// Simple error and debug definitions


#ifndef DEBUG_H
#define DEBUG_H


#include <iostream>


/* Macro definition:
********************/

/*
 * Standard error is used to transfer error codes from
 * the application to the shell
 */
#define ERROR(x) std::cerr << x << std::endl

// Debug macro definition
#ifdef USEDEBUG
#define DEBUG(x) std::cout << x << std::endl
#else
#define DEBUG(x) 
#endif

#endif
