// A small time measurement class

#ifndef HW_TIME_HPP
#define HW_TIME_HPP

#include "sqj2_tb.hpp"

#if defined (__SDSCC__)
	class stopwatch
	{
		public:
			uint64_t t1, t2;
			double cpu_p;
			stopwatch() { cpu_p = 1.0 / (double) sds_clock_frequency(); };
			inline void start() { t1 = sds_clock_counter(); };
			inline void stop() { t2 = sds_clock_counter(); };
			inline double duration() { return ( (t2 - t1)*cpu_p ); };
	};
#else
	class stopwatch
	{
		public:
			std::chrono::high_resolution_clock::time_point t1, t2;
			stopwatch() {};
			inline void start() { t1 = std::chrono::high_resolution_clock::now(); };
			inline void stop() { t2 = std::chrono::high_resolution_clock::now(); };
			inline double duration() { return std::chrono::duration_cast< std::chrono::duration<double> >(t2 - t1).count(); };
	};
#endif

#endif
