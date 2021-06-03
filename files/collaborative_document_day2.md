# Collaborative Document. Day 2, 3 June

2021-06-03-gpu

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](https://hackmd.io/AjdithOLQx2f26vpdcZcuA)

Collaborative Document day 1: [link](https://hackmd.io/Grw3zJBdQMmzJUOL_1GkNQ)

Collaborative Document day 2: [link](https://hackmd.io/AjdithOLQx2f26vpdcZcuA)

## Zoom link


## 👮Code of Conduct

* Participants are expected to follow those guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.

In case of violations contact the instructors, if you have troubles with an instructor please write to training@esciencecenter.nl 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, type `/hand` in the chat window.

To get help, type `/help` in the chat window.

You can ask questions in the document or chat window and helpers will try to help you.

## 🖥 Workshop website

* [Course](https://escience-academy.github.io/2021-06-02-gpu/)
* [JupyterHub](https://jupyter.lisa.surfsara.nl/jhlsrf007)
* [Google Colab](https://colab.research.google.com)
* [Post-workshop survey](https://www.surveymonkey.com/r/8RTGKDR)

🛠 Setup

* [JupyterHub documentation](https://servicedesk.surfsara.nl/wiki/display/WIKI/User+Manual+-+Student)

## 👩‍🏫👩‍💻🎓 Instructors

Alessio Sclocco, Hanno Spreeuw

## 🧑‍🙋 Helpers

Jens Wehner

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call

Name / pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city

## 🗓️ Agenda

09:00 	Welcome and icebreaker

09:15 	Introduction to CUDA

10:15 	Coffee break

10:30 	CUDA memories and their use

11:30 	Coffee break

11:45 	Data sharing and synchronization

12:45 	Wrap-up and post-workshop survey

13:00 	END

## 🧠 Collaborative Notes

### Icebreaker

If you could switch lives with anyone for a day, who would it be?

* I want to switch one day with an astronaut, because I want to see earth from space
* My wife (haha nice one!) :+1: 
* My kid 
* Bob Ross :lower_left_paintbrush: 
* I would like to switch live one day with my cat, he has an awesome life. Free dinner, sleeping, free live :+1: :cat: 
* 007 (this will make Hanno jealous)
* A bird (or should it be a person ;-) ) Excellent choice, I'd like to fly also
* One of the scientist who is currently floating in the ISS. :+1:
* James Bond, then I have a drivers licence
* Rafael Nadal
* Alex Thompson (solo ocean sailor)
* A GPU expert :laughing: 
* Myself when I was younger, just one day in lazy highschool class would be great!
* Rutte (to know what it's like to have no 'visie') :100: 
* Linus Torvalds :+1: 
*  an astronaut to go to space for a day
## Command log

#### jupyter lab
Select `Inside Course Hours` when starting the jupyterhub server

#### Google colab
Otherwise you can use [google collab](https://colab.research.google.com) 

When using google colab use `Runtime -> hardware accelerator -> GPU` to activate the GPU

and then type `!pip install cupy-cuda110` into the first box

#### Creating a new notebook

name: GPU_CUDA_Day2

Adding two vectors
```python=
def vector_add(A,B,C,size):
    for item in range(0,size):
        C[item]=A[item]+B[item]
    return C
```

We use CUDA for device programming because although it is completely proprietary it is still the most common. CUDA can relatively easily converted to HIP for AMD GPUs and DPC++ for Intel GPUs


```cpp=
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
```
```cpp=
int item = threadIdx.x;
C[item] = A[item] + B[item];
```

This is the part of the code in which we do the actual work. As you may see, it looks similar to the innermost loop of our `vector_add` Python function, with the main difference being in how the value of the `item` variable is evaluated.

In fact, while in Python the content of `item` is the result of the `range` function, in CUDA we are reading a special variable, i.e. `threadIdx`, containing a triplet that indicates the id of a thread inside a three-dimensional CUDA block. In this particular case we are working on a one dimensional vector, and therefore only interested in the first dimension, that is stored in the `x` field of this variable.

```python=
import cupy

# size of the vectors
size = 1024

# allocating and populating the vectors
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)

# CUDA vector_add
vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
'''
vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, "vector_add")

#first bracket is how many blocks second bracket threads per block
vector_add_gpu((1, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))
```

To check results
```python=
import numpy

a_cpu = cupy.asnumpy(a_gpu)
b_cpu = cupy.asnumpy(b_gpu)
c_cpu = numpy.zeros(size, dtype=numpy.float32)
vector_add(a_cpu, b_cpu, c_cpu, size)

# test
if numpy.allclose(c_cpu, c_gpu):
    print("Correct results!")
```

If you change `size=2048` you will get an error because CUDA limits the number of threads on a single block. 

To go back to our example, we can modify che grid specification from `(1, 1, 1)` to `(2, 1, 1)`, and the block specification from `(size, 1, 1)` to `(size // 2, 1, 1)`. If we run the code again, we should now get the expected output.

We already introduced the special variable `threadIdx` when introducing the `vector_add` CUDA code, and we said it contains a triplet specifying the coordinates of a thread in a thread block. CUDA has other variables that are important to understand the coordinates of each thread and block in the overall structure of the computation.

These special variables are `blockDim`, `blockIdx`, and `gridDim`, and they are all triplets. The triplet contained in `blockDim` represents the size of the calling thread’s block in three dimensions. While the content of `threadIdx` is different for each thread in the same block, the content of `blockDim` is the same because the size of the block is the same for all threads. The coordinates of a block in the computational grid are contained in `blockIdx`, therefore the content of this variable will be the same for all threads in the same block, but different for threads in different blocks. Finally, `gridDim` contains the size of the grid in three dimensions, and it is again the same for all threads.

```
threadIdx <-Index of a thread in a block (3d)
blockIdx <- Index of a block in a gird (3d)
blockDim <- Size of a block, i.e number of threads (3d)
gridDim <- Size of the grid (3d)
```

<img src="https://upload.wikimedia.org/wikipedia/commons/5/5b/Block-thread.svg">


To make the code work for arbitrary size:

```cpp=
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   if ( item < size )
   {
      C[item] = A[item] + B[item];
   }
}
```

```python=
import math

threads_per_block= 32
grid_size=(int(math.ceil(size/threads_per_block)),1,1)
block_size = (threads_per_block,1,1)

vector_add_gpu(grid_size, block_size, (a_gpu, b_gpu, c_gpu, size))
```

#### Sweet, sweet memories

Registers, Global, and Local Memory

<img src="https://www.3dgep.com/wp-content/uploads/2011/11/CUDA-memory-model.gif">

##### Registers

Registers are fast on-chip memories that are used to store operands for the operations executed by the computing cores.

Did we encounter registers in the `vector_add` code used in the previous episode? Yes we did! The variable `item` is, in fact, stored in a register for at least part, if not all, of a thread’s execution. In general all scalar variables defined in CUDA code are stored in registers.

Registers are local to a thread, and each thread has exclusive access to its own registers: values in registers cannot be accessed by other threads, even from the same block, and are not available for the host. Registers are also not permanent, therefore data stored in registers is only available during the execution of a thread.

If we want to make registers use more explicit in the `vector_add` code, we can try to rewrite it in a slightly different, but equivalent, way.
```cpp=
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   float temp_a, temp_b, temp_c;

   if ( item < size )
   {
       temp_a = A[item];
       temp_b = B[item];
       temp_c = temp_a + temp_b;
       C[item] = temp_c;
   }
}
```
This it totally unnecessary in the case of our example, because the compiler will determine on its own the right amount of registers to allocate per thread, and what to store in them. However, explicit register usage can be important for reusing items already loaded from memory.

The same with arrays
```cpp=
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   float temp[3];

   if ( item < size )
   {
       temp[0] = A[item];
       temp[1] = B[item];
       temp[2] = temp[0] + temp[1];
       C[item] = temp[2];
   }
}
```

##### Global Memory

Global memory can be considered the main memory space of the GPU in CUDA. It is allocated, and managed, by the host, and it is accessible to both the host and the GPU, and for this reason the global memory space can be used to exchange data between the two. It is the largest memory space available, and therefore it can contain much more data than registers, but it is also slower to access. This memory space does not require any special memory space identifier.

Memory allocated on the host, and passed as a parameter to a kernel, is by default allocated in global memory.

Global memory is accessible by all threads, from all thread blocks. This means that a thread can read and write any value in global memory.


##### Local Memory

Memory can also be statically allocated from within a kernel, and according to the CUDA programming model such memory will not be global but local memory. Local memory is only visible, and therefore accessible, by the thread allocating it. So all threads executing a kernel will have their own privately allocated local memory.

Local memory is not not a particularly fast memory, and in fact it has similar throughput and latency of global memory, but it is much larger than registers. As an example, local memory is automatically used by the CUDA compiler to store spilled registers, i.e. to temporarily store variables that cannot be kept in registers anymore because there is not enough space in the register file, but that will be used again in the future and so cannot be erased.

##### Constant memory

Constant memory is a read only cache which content can be broadcasted to multiple threads in a block. It is allocated by the host using the `__constant__` identifier, and it must be a global variable, i.e. it must be declared in a scope that contains the kernel, not inside the kernel. Although constant memory is declared on the host, it is not accessible by the host itself.

```cpp=
extern "C" {
#define BLOCKS 2

__constant__ float factors[BLOCKS];

__global__ void sum_and_multiply(const float * A, const float * B, float * C, const int size)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    C[item] = (A[item] + B[item]) * factors[blockIdx.x];
}
}
```

in python
```python=
# size of the vectors
size = 2048

# allocating and populating the vectors
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)
# prepare arguments
args = (a_gpu, b_gpu, c_gpu, size)

# CUDA code
cuda_code = r'''
extern "C" {
#define BLOCKS 2

__constant__ float factors[BLOCKS];

__global__ void sum_and_multiply(const float * A, const float * B, float * C, const int size)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    C[item] = (A[item] + B[item]) * factors[blockIdx.x];
}
}
'''

# Compile and access the code
module = cupy.RawModule(code=cuda_code)
sum_and_multiply = module.get_function("sum_and_multiply")
# Allocate and copy constant memory
factors_ptr = module.get_global("factors")
factors_gpu = cupy.ndarray(2, cupy.float32, factors_ptr)
factors_gpu[...] = cupy.random.random(2, dtype=cupy.float32)

sum_and_multiply((2, 1, 1), (size // 2, 1, 1), args)

```

##### Shared memory

Shared memory is a CUDA memory space that is shared by all threads in a thread block. In this case shared means that all threads in a thread block can write and read to block-allocated shared memory, and all changes to this memory will be eventually available to all threads in the block.
To allocate an array in shared memory we need to preface the definition with the identifier `__shared__`.

```cpp=
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
  int item = (blockIdx.x * blockDim.x) + threadIdx.x;
  int offset = threadIdx.x * 3;
  extern __shared__ float temp[];

  if ( item < size )
  {
      temp[offset + 0] = A[item];
      temp[offset + 1] = B[item];
      temp[offset + 2] = temp[offset + 0] + temp[offset + 1];
      C[item] = temp[offset + 2];
  }
}
```

```python=
size = 2048

a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)
gpu_args = (a_gpu, b_gpu, c_gpu, size)

a_cpu = cupy.asnumpy(a_gpu)
b_cpu = cupy.asnumpy(b_gpu)
c_cpu = numpy.zeros(size, dtype=numpy.float32)

vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
  int item = (blockIdx.x * blockDim.x) + threadIdx.x;
  int offset = threadIdx.x * 3;
  extern __shared__ float temp[];

  if ( item < size )
  {
      temp[offset + 0] = A[item];
      temp[offset + 1] = B[item];
      temp[offset + 2] = temp[offset + 0] + temp[offset + 1];
      C[item] = temp[offset + 2];
  }
}
'''
vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, "vector_add")

threads_per_block = 32
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)
vector_add_gpu(grid_size, block_size, gpu_args, shared_mem=(threads_per_block * 3 * cupy.dtype(cupy.float32).itemsize))
vector_add(a_cpu, b_cpu, c_cpu, size)

numpy.allclose(c_cpu, c_gpu)

```

The code is now correct, although it is still not very useful. We are definitely using shared memory, and we are using it the correct way, but there is no performance gain we achieved by doing so. In practice, we are making our code slower, not faster, because shared memory is slower than registers.

```python=
def histogram(input_array, output_array):
    for item in input_array:
        output_array[item] = output_array[item] + 1
    return output_array

```

```python=
size=2048
input_gpu = cupy.random.randint(256,size=size,dtype=cupy.int32)
input_cpu=cupy.asnumpy(input_gpu)

output_gpu =cupy.zeros(256,dtype=cupy.int32)
output_cpu=cupy.asnumpy(output_gpu)

histogram_cuda_code = r'''
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    extern __shared__ int temp_histogram[];
    //Set temp_histogram entries to zero
    temp_histogram[threadIdx.x]=0;
    __syncthreads();
    atomicAdd(&(temp_histogram[input[item]]), 1);
    // To solve memory error, we need to explicitly synchronize all threads in a block, 
    __syncthreads();
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
'''
histogram_gpu = cupy.RawKernel(histogram_cuda_code, "histogram")

threads_per_block = 256
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

histogram_gpu(grid_size, block_size, (input_gpu, output_gpu),shared_mem=(threads_per_block* cupy.dtype(cupy.int32)))
histogram(input_cpu, output_cpu)

numpy.allclose(output_cpu, output_gpu)

```

## Exercises

### Challenge

We know enough now to pause for a moment and do a little exercise. Assume that in our `vector_add` kernel we replace the following line:

```
int item = threadIdx.x;
```

With this other line of code:

```
int item = 1;
```

What will the result of this change be?

1) Nothing changes
2) Only the first thread is working
3) Only `C[1]` is written
4) All elements of `C` are zero

#### Answers

#### Solution

The correct answer is number 3, only the element `C[1]` is written, and we do not even know by which thread!

### Challenge

In the following code, fill in the blank to work with vectors that are larger than the largest CUDA block (i.e. 1024).

```
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = ______________;
   C[item] = A[item] + B[item];
}
```

#### Solution

The correct answer is `(blockIdx.x * blockDim.x) + threadIdx.x`.

```
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   C[item] = A[item] + B[item];
}
```

## Open Questions

Here you can type questions that you want answered in the document, or at some point during the workshop.

* Work related: I often have to transpose very large .h5 files shape =(millions, by half million).  Would you recommend to do this in chuncks on gpu's?
    * Good question, as you said you cannot transpose the whole thing at once, so you will have a lot of transfers. Can be faster, but not necessarily. Depends on your graphics card(s) and CPU. 
*

## Feedback

### What went well :+1:

* Nice to have a GPU master for a teacher :+1: :+1: :+1:
* Very useful information
* Very nice course today! Good pace and good example code
* Very good course, thank you!
* CUDA programming was very tough but also very interesting. I don't think it can be explained in an easier manner than Alessio did just now.

### What could be improved :-1:
* It would be nice if blocks and threads, and their dimensions, were explained before we define their sizes :+1:
* Quick explanation slide about registers and memory types would be nice. I could use a mental model :+1: :+1:
* I found the last hour almost impossible to follow, which is a pity because the rest was great
* Type along: may be a good idea, but if you drop out/make a typo you don't find immediately, AND there is no working code in shared document to get back on track quickly, it brings you only frustration. Trying to find/fix your bug AND listening to the content of the course is IMPOSSIBLE.
* Please avoid copy & paste (it is easy to get lost at those moments)
* Include some profiling (timing)
* Some real life problems to sove using GPUs would be good to have to get a feeling how they can be solved using GPUs (+ solutions)

## 📚 Resources

* [Post-workshop survey](https://www.surveymonkey.com/r/8RTGKDR)
* [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
* [CUDA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* [Upcoming workshops](https://escience-academy.github.io/)
