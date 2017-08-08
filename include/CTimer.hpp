/*
 * CTimer.hpp
 *
 *  Created on: 8 ago. 2017
 *      Author: juanitov
 */

#ifndef INCLUDE_CTIMER_HPP_
#define INCLUDE_CTIMER_HPP_

#include <ctime>
#include <chrono>

////////////////////////////////////////////////////////////////////////////////
class CTimer
{
	public:

		CTimer()
		{
			start_ = std::chrono::high_resolution_clock::time_point::min();
			end_ = std::chrono::high_resolution_clock::time_point::min();
		}

		inline void tic()
		{
			start_ = std::chrono::high_resolution_clock::now();
		}

		inline void tac()
		{
			end_ = std::chrono::high_resolution_clock::now();
		}

		inline double Elasped()
		{
			auto duration =
				std::chrono::duration_cast<std::chrono::milliseconds>(
																end_ - start_);
			return duration.count();
		}

	private:

		std::chrono::high_resolution_clock::time_point start_;
		std::chrono::high_resolution_clock::time_point end_;
};

#endif /* INCLUDE_CTIMER_HPP_ */
