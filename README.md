# matrix-cuda
Matrix multiplication in CUDA, this is a toy program for learning CUDA, some functions are reusable in other project


# Test Results
#### The following tests were carried out on a GTX 660 card

```text
C:\Users\Dakota\Documents\src\matrix-cuda>x64\Release\matrix_cuda.exe
==================================================================================
Enter m, n, and k: 1024 1024 1024

----------------------------------------------------------------------------------
Matrix type: int
Time elapsed on matrix multiplication of 1024x1024 . 1024x1024 on GPU: 17.4332ms.

----------------------------------------------------------------------------------
Matrix type: float
Time elapsed on matrix multiplication of 1024x1024 . 1024x1024 on GPU: 17.5297ms.

----------------------------------------------------------------------------------
Matrix type: double
Time elapsed on matrix multiplication of 1024x1024 . 1024x1024 on GPU: 30.9786ms.

==================================================================================

C:\Users\Dakota\Documents\src\matrix-cuda>x64\Release\matrix_cuda.exe
==================================================================================
Enter m, n, and k: 2048 2048 2048

----------------------------------------------------------------------------------
Matrix type: int
Time elapsed on matrix multiplication of 2048x2048 . 2048x2048 on GPU: 132.073ms.

----------------------------------------------------------------------------------
Matrix type: float
Time elapsed on matrix multiplication of 2048x2048 . 2048x2048 on GPU: 128.334ms.

----------------------------------------------------------------------------------
Matrix type: double
Time elapsed on matrix multiplication of 2048x2048 . 2048x2048 on GPU: 209.006ms.

==================================================================================

C:\Users\Dakota\Documents\src\matrix-cuda>x64\Release\matrix_cuda.exe
==================================================================================
Enter m, n, and k: 4096 4096 4096

----------------------------------------------------------------------------------
Matrix type: int
Time elapsed on matrix multiplication of 4096x4096 . 4096x4096 on GPU: 950.203ms.

----------------------------------------------------------------------------------
Matrix type: float
Time elapsed on matrix multiplication of 4096x4096 . 4096x4096 on GPU: 948.593ms.

----------------------------------------------------------------------------------
Matrix type: double
Time elapsed on matrix multiplication of 4096x4096 . 4096x4096 on GPU: 2630.69ms.

==================================================================================
```

