#include <thread>
#include <deque>
#include <condition_variable>
#include <atomic>
#include <vector>

#include "../iplib/parallel.h"

namespace ip
{
	struct ParallelThreadTask
	{
		std::function<void()> *func;

	public:
		ParallelThreadTask(std::function<void()> &func)
			: func(&func) {}
	};

	// ======================================================================================================

	class ParallelThread
	{
	private:
		int index;
		std::thread thr;
		std::deque<ParallelThreadTask*> tasks;
		ParallelThreadTask *current;
		std::mutex execution_sync, queue_sync;
		std::condition_variable queue_var;

	public:
		ParallelThread(int index)
			: current(nullptr), index(index)
		{
			thr.swap(std::thread(&ParallelThread::ThreadProc, this));
			thr.detach();
			// thread(&ThreadProc, this).detach();
		}

	private:
		void AddEmptyTask()
		{
			std::unique_lock<std::mutex> lock(queue_sync);
			tasks.push_back(nullptr);
			queue_var.notify_one();
		}

	public:
		void StopThread()
		{
			AddEmptyTask();
			thr.join();
			// printf("~ParallelThread #%d\n", index);
		}

		ParallelThreadTask* AddTask(std::function<void()> &func)
		{
			ParallelThreadTask *task = new ParallelThreadTask(func);

			std::unique_lock<std::mutex> lock(queue_sync);
			tasks.push_back(task);
			// printf("[AddTask %d]: Add 0x%x (func = 0x%x)\n", index, task, &func);
			queue_var.notify_one();

			return task;
		}

		void CompleteTask(ParallelThreadTask *task)
		{
			std::unique_lock<std::mutex> lock(queue_sync);
			std::unique_lock<std::mutex> ex_lock(execution_sync, std::defer_lock);

			if (current == task)
				ex_lock.lock();

			if (task->func != nullptr)
			{
				// printf("[CompleteTask %d]: Deactivate task 0x%x (func = 0x%x)\n", index, task, task->func);
				task->func = nullptr;
			}
			else
			{
				// printf("[CompleteTask %d]: Delete task 0x%x\n", index, task);
				delete task;
			}
		}

	private:

		bool GetNext()
		{
			std::unique_lock<std::mutex> lock(queue_sync);

			while (true)
			{
				current = nullptr;

				while (tasks.empty())
					queue_var.wait(lock);

				current = tasks.front();
				tasks.pop_front();

				if (current == nullptr)
					return false;

				if (current->func != nullptr)
					return true;

				// printf("[Worker %d]: Delete task 0x%x\n", index, current);
				delete current;
			}
		}

		void ExecuteNext()
		{
			std::unique_lock<std::mutex> lock(execution_sync);

			// printf("[Worker %d]: Execute task 0x%x (func = 0x%x)\n", index, current, current->func);

			if (current->func != nullptr)
			{
				current->func->operator()();

				// printf("[Worker %d]: Mark task as completed 0x%x (func = 0x%x)\n", index, current, current->func);
				current->func = nullptr;
			}
			else
				delete current;

		}

		void ThreadProc()
		{
			while (GetNext())
				ExecuteNext();
		}
	};

	// ======================================================================================================

	class ParallelHost
	{
		std::vector<std::unique_ptr<ParallelThread>> threads;

	public:
		ParallelHost()
		{
			unsigned int proc_count = std::thread::hardware_concurrency();
			// printf("Number of processors = %d\n", proc_count);

			for (int i = 0; i < (int)proc_count - 1; i++)
				threads.emplace_back(new ParallelThread(i));
		}

		/* ~ParallelHost()
		{
			for (auto &thr : threads)
				thr->StopThread();
		} */

		void Do(std::function<void()> func)
		{
			std::vector<ParallelThreadTask*> tasks;
			tasks.reserve(threads.size());

			for (auto &thr : threads)
			{
				tasks.push_back(thr->AddTask(func));
			}

			func();

			for (int i = (int)tasks.size() - 1; i >= 0; i--)
			{
				ParallelThreadTask *task = tasks[i];
				threads[i]->CompleteTask(task);
			}
		}

		struct AggregateData
		{
			ParallelThreadTask* task;
			std::function<void()> func;
			void* state;
			bool hasValue;
		};

		void Aggregate(std::function<void(void* state)> func, std::function<void(void* accumulator, void* state)> aggregator, void* target, size_t state_size)
		{
			std::vector<AggregateData> tasks(threads.size());
			std::vector<char> buffer(state_size * threads.size());

			for (size_t i = 0; i < threads.size(); i++)
			{
				auto &task = tasks[i];

				task.state = &buffer[i * state_size];
				task.hasValue = false;

				task.func = [&func, &task]()
				{
					task.hasValue = true;
					func(task.state);
				};

				task.task = threads[i]->AddTask(task.func);
			}

			func(target);

			for (size_t i = 0; i < threads.size(); i++)
			{
				auto& task = tasks[i];

				threads[i]->CompleteTask(task.task);

				if (task.hasValue)
					aggregator(target, task.state);
			}
		}

		void For(std::function<void(std::atomic_int& counter)> func, int initial = 0)
		{
			std::atomic_int counter;
			counter.store(initial);

			Do([&counter, &func]()
			{
				func(counter);
			});
		}

		void For(int beginInclusive, int endExclusive, std::function<void(int y)> func)
		{
			std::atomic_int counter;
			counter.store(beginInclusive);

			Do([&counter, &func, endExclusive]()
			{
				for (int y = counter++; y < endExclusive; y = counter++)
				{
					func(y);
				}
			});
		}
	};

	// ======================================================================================================

	std::recursive_mutex mutex;
	std::unique_ptr<ParallelHost> parallel;

	ParallelHost& GetParallelHost()
	{
		if (!parallel)
		{
			std::unique_lock<std::recursive_mutex> lock_guard(mutex);

			if (!parallel)
				parallel.reset(new ParallelHost());
		}

		return *parallel.get();
	}

	// ======================================================================================================

	void Parallel::Do(std::function<void()> func)
	{
		GetParallelHost().Do(func);
	}

	void Parallel::For(std::function<void(std::atomic_int &counter)> func, int initial)
	{
		GetParallelHost().For(func, initial);
	}

	void Parallel::For(int beginInclusive, int endExclusive, std::function<void(int y)> func)
	{
		GetParallelHost().For(beginInclusive, endExclusive, func);
	}

	void Parallel::Reset()
	{
		std::unique_lock<std::recursive_mutex> lock_guard(mutex);

		if (parallel)
		{
			parallel.reset();
		}
	}

	void Parallel::Aggregate(std::function<void(void* state)> func, std::function<void(void* accumulator, void* state)> aggregator, void* target, size_t state_size)
	{
		GetParallelHost().Aggregate(func, aggregator, target, state_size);
	}

	// ======================================================================================================
}