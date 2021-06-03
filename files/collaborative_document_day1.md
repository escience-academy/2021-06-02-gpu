# Collaborative Document. Day 1, 2 June

2021-06-02-gpu

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](https://hackmd.io/Grw3zJBdQMmzJUOL_1GkNQ)

Collaborative Document day 1: [link](https://hackmd.io/Grw3zJBdQMmzJUOL_1GkNQ)

Collaborative Document day 2: [link](https://hackmd.io/AjdithOLQx2f26vpdcZcuA)

## 👮Code of Conduct

* Participants are expected to follow those guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
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

🛠 Setup

* [JupyterHub documentation](https://servicedesk.surfsara.nl/wiki/display/WIKI/User+Manual+-+Student)

## 👩‍🏫👩‍💻🎓 Instructors

Alessio Sclocco, Hanno Spreeuw

## 🧑‍🙋 Helpers

Jens Wehner

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call

Name / pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city


## 🗓️ Agenda

09:00 -	Welcome and icebreaker
09:15 -	Introduction
09:30 -	Convolve an image with a kernel on a GPU using CuPy
10:15 -	Coffee break
10:30 -	Running CPU/GPU agnostic code using CuPy
11:30 -	Coffee break
11:45 -	Run your Python code on a GPU using Numba
12:45 -	Wrap-up
13:00 -	END

## 🧠 Collaborative Notes

### Icebreaker

If you could only eat one meal for the rest of your life, what will it be and why?
* cheese fondue to end it quickly
* Sachertorte
* Risotto
* Moussaka
* Thai food from the Bird snackbar in Amsterdam (if I'm limitted to what is avaiable for me here at the moment)
* Pizze quattro stagioni - tasty and it has many ingredients, will keep you alive! :pizza: 
* ramen soup (is tasty)
* Tuna melts
* gegratineerde aardappelen :potato: 
* Sushi :+1: :fish:
* Broccoli with sour cream and anchovies (thanks for the dinner inspo!!)
* pasta
* noodle soup (have different combinations and nutritious enough)
* vegetarian pizza, all the major food groups and taste!
* My mother's chicken
* Paella
* Pizza, b/c you can still have many variations of Pizza :+1: :+1: 
* Argentinian Asado, 
* Wienerschnitzel, because I want to go to Wien (while you're at it, have a Sachertorte there) (okay, will do!!:-)
* Stroganoff (Russian dish, surprisingly famous in Brazil)
* Lasagna
* Patat met :+1:
* Paella :rice_scene: :+1: 
* Lunch
* Tempeh with peanut butter
* Mixed fried fish (very tasty) 
* Bacalhau à brás
* Császármorzsa :pancakes: 

### Zoom Link


### Exercise 1

Compute the speedup for the convolution adding also transfer time to and from the GPU.

### Exercise 2

Find all prime numbers using `find_all_primes` but replacing the innermost loop with the new `check_prime_gpu_kernel` code and report the speedup.

## Command log

### Setup

```
# If using Google Colab you may need to install CuPy and Numba
!pip install cupy
!pip install numba
```

```
import numpy as np
import cupy as cp
import numba as nb
```

### Convolution

```
primary_unit = np.zeros((16, 16))
primary_unit[8, 8] = 1
deltas = np.tile(primary_unit, (128, 128))
%matplotlib inline
import pylab as pyl
pyl.imshow(deltas[0:64, 0:64])
pyl.show()
```

```
x, y = np.meshgrid(np.linspace(-2, 2, 16), np.linspace(-2, 2, 16))
distsq = x**2 + y**2
gauss = np.exp(-distsq)
pyl.imshow(gauss)
pyl.show()
```

```
from scipi.signal import convolve2d as convolve2d_cpu
convolved_image_on_cpu = convolve2d_cpu(deltas, gauss)
pyl.imshow(convolved_image_on_cpu[0:64, 0:64])
pyl.show()
```

```
# Time the computation on the CPU
%timeit convolve2d_cpu(deltas, gauss)
```

```
# Copy data to the GPU and execute convolution on the GPU
deltas_gpu = cp.asarray(deltas)
gauss_gpu = cp.asarray(gauss)
from cupyx.scipy.signal import convolve2d as convolve2d_gpu
image_convolved_on_gpu = convolve2d_gpu(deltas_gpu, gauss_gpu)
```

```
# Time the computation on the GPU
%timeit convolve2d_gpu(deltas_gpu, gauss_gpu)
# Compute speedup over CPU
speedup = (5230 / 13.8)
print(speedup)
```

```
np.allclose(convolved_image_on_cpu, image_convolved_on_gpu)
```

```
# Transfer results back to CPU and plot
image_convolved_on_gpu_copied_back_to_cpu = cp.asnumpy(image_convolved_on_gpu)
pyl.imshow(image_convolved_on_gpu_copied_back_to_cpu[0:64, 0:64])
pyl.show()
```

```
# Error
convolve2d_gpu(deltas, gauss)
```

```
# Time a whole Jupyter cell
%%timeit
deltas_gpu = cp.asarray(deltas)
gauss_gpu = cp.asarray(gauss)
image_convolved_on_gpu = convolve2d_gpu(deltas_gpu, gauss_gpu)
image_convolved_on_gpu_copied_back_to_cpu = cp.asnumpy(image_convolved_on_gpu)
```

### 1D convolution

```
deltas_1d = np.ravel(deltas)
gauss_1d = np.diagonal(gauss)
one_dimensional_convolution_on_cpu = np.convolve(deltas_1d, gauss_1d)
```

```
deltas_1d_gpu = cp.asarray(deltas_1d)
gauss_1d_gpu = cp.asarray(gauss_1d)
np.convolve(deltas_1d_gpu, gauss_1d_gpu)
```

### Numba and prime numbers

```
# Function to find all prime numbers up to upper
def find_all_primes_cpu(upper):
    all_prime_numbers=[]
    for num in range(2, upper):
        # all prime numbers are greater than 1
        for i in range(2, num):
            if (num % i) == 0:
                break
        else:
            all_prime_numbers.append(num)
    return all_prime_numbers
```

```
find_all_primes(100)
```

```
%timeit -n 1 find_all_primes(10000)
```

```
# Increase performance on the CPU using the Numba just-in-time compiler
from numba import jit

find_all_primes_fast = jit(nopython=True)(find_all_primes)
%timeit -n 1 find_all_primes_fast(10000)
```

```
# Using CUDA (GPU) from Numba
from numba import cuda

@cuda.jit
def check_prime_gpu_kernel(num, result):
    result[0] = 0
    for i in range(2, num):
        if (num % i) == 0:
            break
    else:
        result[0] = num

result = np.zeros((1), np.int32)
check_prime_gpu_kernel[1, 1](10, result)
print(result[0])
check_prime_gpu_kernel[1, 1](11, result)
print(result[0])
check_prime_gpu_kernel[1, 1](12, result)
print(result[0])
```

```
def find_all_primes_cpu_and_gpu(upper):
    all_primes = []
    for num in range(2, upper):
        result = np.zeros((1), np.int32)
        check_prime_gpu_kernel[1, 1](num, result)
        if result[0] > 0:
            all_primes.append(num)
    return all_primes

find_all_primes_cpu_and_gpu(100)
cpu_and_gpu = np.array(find_all_primes_cpu_and_gpu(100))
just_cpu = np.array(find_all_primes(100))
np.allclose(just_cpu, cpu_and_gpu)
```

```
# Expected slowdown
%timeit -n 1 find_all_primes_cpu_and_gpu(10000)
```

```
check_these_numbers_for_primes = np.arange(2, 10000, dtype=np.int32)

@nb.vectorize(["int32(int32)"], target="cuda")
def check_prime_gpu(num):
    for i in range(2, num):
        if (num % i) == 0:
            return 0
    else:
        return num

check_prime_gpu(check_these_numbers_for_primes)
```

```
%timeit check_prime_gpu(check_these_numbers_for_primes)
```

## Open Questions

Here you can type questions that you want answered in the document, or at some point during the workshop.

* If you transfered something to the GPU it stays on the gpu. Do we need to manually release it (if it is not a function variable?). 
    * In this case we are using Python and CuPy, so memory life is determined by Python and the garbage collector
* What is the best way to release it? just variable = []?
    * I would say that it should work in telling the garbage collector that it is not anymore necessary to have that memory allocated
        * So del my_object, gc.collect()?
            * [Documentation on memory](https://docs.cupy.dev/en/stable/user_guide/memory.html)
* Can you see (within python) how much RAM the GPU has left?
    * The CuPy interface should give you access to low level CUDA primitives to work with the GPU memory
    * Or you can use nvidia-smi from the command line as Hanno showed

## Feedback

### What went well :+1:

* Jupyter hub works nicely :+1: :+1:
* The excercises where nice
* Pace was great, not too slow or too fast :+1: :+1: 
* Good examples to code along with :+1: :+1: :+1:
* Nice that COLAB was a backup option :+1:
* Jupyter Hub on LISA was great :+1:
* nice examples and good pace :+1:
* Nice to have collaborateive document
* Great resources
* Nice!!!
* Good introductory examples, looking forward to tomorrow
* Good amount of people helping with the courses besides the instructors! :+1: :+1: 
* Feedback from the Helper is very in time, nice work! "very in time" -> just in time? :-)
* Nice to code along instead of looking at slides

### What could be improved :-1:

* a list of pro's and cons for each method
* it would be easier for people to recap if there is a guideline in markdown on e.g. what is the task, a small summary...
* You can make the code for primes even more similar for cpu / numba / cuda, better to compare
* maybe quick recap after each method
* more explanation on @nb.vectorize
* JupyterHub worked first, then suddenly disconnected and stopped working. Don't even know if error is on my side, or just crappy server. (Errors that only occur for some participants are always demotivating for those, as they keep trying to find what they did wrong, while the others just say: what are you on about, works for me!)
* faster introduction
* introduction could cover more examples of applications when GPU are a preferable choice
* Have a system in place to help people with technical difficulties, so it does not take up too much time of the course.
* may a bit more in-depth explanation of how gpu does the computation :+1: :+1: :+1:
* Having, code-snippets ready would probably speed up the workshop instead of typing them largely from memory.
* The pace was super slow, too slow.
* Make a more detailed setting-up guide. We spend too much time today trying to fix Jupyter/logging in for some people. 
* It might have been more efficient to work with an existing notebook, and spend more time on exercises. Now we typed a lot of things over.
* Give a schematic overview of each method. I.e. pure-python does things sequential, numba.jit also does it sequential but now with C/C++ code behind the scenes, and when using cuda (??? not sure what happens when we did this). :+1: :+1: 
* JupyterHub stopped at 1pm spot-on (end of 'course-hours'). It would be helpful to extend the 'course-hours' for e.g. 10 minutes to be able to export your notebook.


## 📚 Resources

* [upcoming eScience Center courses](https://escience-academy.github.io/)
* [NL-RSE community, the Dutch community for Research Software Engineers](www.nl-rse.org)
* [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
* [Numba User Guide](https://numba.pydata.org/numba-doc/latest/user/index.html)
* [Good overview of jit](https://medium.com/starschema-blog/jit-fast-supercharge-tensor-processing-in-python-with-jit-compilation-47598de6ee96)
* [tips for python](https://book.pythontips.com/en/latest/)

Per request the `prime`-example implemented in Rust, with the addition of an optimization of the `for`-loop (`sqrt()`):
```rust
fn main() {

let mut all_primes: Vec<u64> = Vec::new();

for num in 2u64..10000 {
    let  stop = ((num as f64).sqrt() + 1.0) as u64; 
    let mut found = true;
    for i in 2..stop {
        if num%i == 0 {
            found = false;
            break;
        }
    }
    if found {
        all_primes.push(num);
    }
}

println!("{:?}", all_primes.len());

}
```

