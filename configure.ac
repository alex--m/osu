#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.59])
AC_INIT([OSU-Micro-Benchmarks], [7.0], [mvapich-discuss@cse.ohio-state.edu])
AC_CONFIG_SRCDIR([c/mpi/pt2pt/osu_latency.c])

AM_INIT_AUTOMAKE([foreign])
LT_INIT

dnl Disable C/C++ verions
AC_ARG_ENABLE([c],
              [AS_HELP_STRING([--disable-c],
                              [Disable C benchmarks (default is no).])
              ],
              [enable_c=no],
              [])

dnl
dnl Add NVCCFLAGS for additional flags
dnl
AC_ARG_VAR(NVCCFLAGS,
	[extra NVCCFLAGS used in building OMB with CUDA kernel support])

AC_ARG_ENABLE([openacc],
              [AS_HELP_STRING([--enable-openacc],
                              [Enable OpenACC benchmarks])
              ],
              [],
              [enable_openacc=no])

AC_ARG_ENABLE([cuda],
              [AS_HELP_STRING([--enable-cuda],
			      [Enable CUDA benchmarks (default is no).  Specify
			       --enable-cuda=basic to enable basic cuda support
			       without using cuda kernel support for
			       non-blocking collectives])
              ],
              [],
              [enable_cuda=no])

AC_ARG_WITH([cuda],
            [AS_HELP_STRING([--with-cuda=@<:@CUDA installation path@:>@],
                            [Provide path to CUDA installation])
            ],
            [AS_CASE([$with_cuda],
                     [yes|no], [],
                     [CPPFLAGS="-I$with_cuda/include $CPPFLAGS"
                      LDFLAGS="-L$with_cuda/lib64 -Wl,-rpath=$with_cuda/lib64 -L$with_cuda/lib -Wl,-rpath=$with_cuda/lib $LDFLAGS"])
            ])

AC_ARG_WITH([cuda-include],
            [AS_HELP_STRING([--with-cuda-include=@<:@CUDA include path@:>@],
                            [Provide path to CUDA include files])
            ],
            [AS_CASE([$with_cuda_include],
                     [yes|no], [],
                     [CPPFLAGS="-I$with_cuda_include $CPPFLAGS"])
            ])

AC_ARG_WITH([cuda-libpath],
            [AS_HELP_STRING([--with-cuda-libpath=@<:@CUDA library path@:>@],
                            [Provide path to CUDA library files])
            ],
            [AS_CASE([$with_cuda_libpath],
                     [yes|no], [],
                     [LDFLAGS="-L$with_cuda_libpath -Wl,-rpath=$with_cuda_libpath $LDFLAGS"])
            ])

AC_ARG_ENABLE([papi],
              [AS_HELP_STRING([--enable-papi],
			      [Enable PAPI support for benchmarks (default is no)])
              ],
              [enable_papi=yes],
              [enable_papi=no])

AC_ARG_WITH([papi],
            [AS_HELP_STRING([--with-papi=@<:@PAPI installation path@:>@],
                            [Provide path to PAPI installation])
            ],
            [AS_CASE([$with_papi],
                     [yes|no], [],
                     [CPPFLAGS="-I$with_papi/include $CPPFLAGS"
                      LDFLAGS="-L$with_papi/lib -Wl,-rpath=$with_papi/lib $LDFLAGS -lpapi"])
            ])


AC_ARG_ENABLE([ncclomb],
            [AS_HELP_STRING([--enable-ncclomb],
                [Enable NCCL benchmarks (default is no). Specify
                --enable-ncclomb to enable NCCL benchmarks support])
            ],
            [],
            [enable_ncclomb=no])

if test "x$enable_ncclomb" = "xyes"; then
    AC_ARG_WITH([nccl],
                [AS_HELP_STRING([--with-nccl=@<:@NCCL installation path@:>@],
                                [Provide path to NCCL installation])
                ],
                [AS_CASE([$with_nccl],
                         [yes|no], [],
                         [CPPFLAGS="-I$with_nccl/include $CPPFLAGS"
                          LDFLAGS="-L$with_nccl/lib -Wl,-rpath=$with_nccl/lib -lnccl $LDFLAGS"])
                ])
fi


AC_ARG_ENABLE([rocm],
              [AS_HELP_STRING([--enable-rocm],
			                  [Enable ROCm benchmarks])
              ],
              [],
              [enable_rocm=no])

AC_ARG_WITH([rocm],
            [AS_HELP_STRING([--with-rocm=@<:@ROCm installation path@:>@],
                            [Provide path to ROCm installation])
            ],
            [AS_CASE([$with_rocm],
                     [yes|no], [],
                     [CPPFLAGS="-I$with_rocm/include $CPPFLAGS"
                      LDFLAGS="-L$with_rocm/lib64 -Wl,-rpath=$with_rocm/lib64 -L$with_rocm/lib -Wl,-rpath=$with_rocm/lib -lamdhip64 $LDFLAGS"])
            ])
AC_ARG_WITH([ucx],
            [AS_HELP_STRING([--with-ucx=@<:@UCX installation path@:>@],
                            [Provide path to UCX installation])
            ],
            [AS_CASE([$with_ucx],
                     [yes|no],
                     [AC_DEFINE([_WITH_UCX_], [1], [UCX available])],
                     [CPPFLAGS="-I$with_ucx/include -I$with_ucx/../src $CPPFLAGS"
                      LDFLAGS="-L$with_ucx/lib64 -Wl,-rpath=$with_ucx/lib64 -L$with_ucx/lib -Wl,-rpath=$with_ucx/lib -lucm -lucs -luct -lucp -lucg $LDFLAGS"])
            ])
AC_ARG_WITH([gnuplot],
            [AS_HELP_STRING([--with-gnuplot=@<:@gnuplot installation path@:>@],
                            [Provide path to gnuplot installation])
            ],
            [AS_IF([test -d $withval],
             [AC_CHECK_PROG(GNUPLOT_CHECK_PATH,gnuplot,yes,no,$with_gnuplot)],
             [AC_MSG_ERROR([Unable to find gnuplot build path. Please check the path.])])])
AS_IF([test x$GNUPLOT_CHECK_PATH = x"yes"],
      [AC_DEFINE_UNQUOTED([_GNUPLOT_BUILD_PATH_],["$with_gnuplot/gnuplot"])],[])
AS_IF([test x$GNUPLOT_CHECK_PATH = x"no"],
      [AC_MSG_ERROR([gnuplot executable not found in the given path.])])
AC_ARG_WITH([convert],
            [AS_HELP_STRING([--with-convert=@<:@ImageMagick installation path@:>@],
                            [Provide path to ImageMagick installation])
            ],
            [AS_IF([test -d $withval],
             [AC_CHECK_PROG(CONVERT_CHECK_PATH,convert,yes,no,$with_convert)],
             [AC_MSG_ERROR([Unable to find ImageMagick build path. Please check the path.])])])
AS_IF([test x$CONVERT_CHECK_PATH = x"yes"],
      [AC_DEFINE_UNQUOTED([_CONVERT_BUILD_PATH_],["$with_convert/convert"])])
AS_IF([test x$CONVERT_CHECK_PATH = x"no"],
      [AC_MSG_ERROR([convert executable not found in the given path.])])
# Checks for programs.
AC_PROG_CC([mpicc oshcc upcc upc++])

# Checks for mpicxx used for compiling kernel.cu in nbc benchmarks and/or the
# upc++ compiler for upcxx benchmarks
AC_PROG_CXX([mpicxx upc++])

# Checks for libraries.
AC_SEARCH_LIBS([sqrt], [m])
AC_SEARCH_LIBS([pthread_join], [pthread])
AC_SEARCH_LIBS([clock_gettime], [rt])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h string.h sys/time.h unistd.h math.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE

# Checks for library functions.
AC_CHECK_FUNCS([getpagesize gettimeofday memset sqrt])

AS_IF([test "x$enable_embedded" = xyes], [
       AS_IF([test x"$enable_mpi3" = xyes], [mpi3_library=true])
       AS_IF([test x"$enable_mpi2" = xyes], [mpi2_library=true])
       AS_IF([test x"$enable_mpi" = xyes], [mpi_library=true])
       AS_IF([test x"$enable_oshm" = xyes], [oshm_library=true])
       AS_IF([test x"$enable_upc" = xyes], [upc_compiler=true])
       AS_IF([test x"$enable_upcxx" = xyes], [upcxx_compiler=true])
       AS_IF([test x"$enable_oshm_13" = xyes], [oshm_13_library=true])
      ], [
       AC_CHECK_FUNC([MPI_Init], [mpi_library=true])
       AC_CHECK_FUNC([MPI_Accumulate], [mpi2_library=true])
       AC_CHECK_FUNC([MPI_Get_accumulate], [mpi3_library=true])
       AC_CHECK_FUNC([shmem_barrier_all], [oshm_library=true])
       AC_CHECK_FUNC([upc_memput], [upc_compiler=true])
       AC_CHECK_DECL([upcxx_alltoall], [upcxx_compiler=true], [],
                     [#include <upcxx.h>])
       AC_CHECK_FUNC([shmem_finalize], [oshm_13_library=true])
      ])

AM_CONDITIONAL([EMBEDDED_BUILD], [test x"$enable_embedded" = xyes])
AM_CONDITIONAL([BUILD_PROFILING_LIB], [test x"$with_plib" = xyes])
AC_SUBST([PMPILIBNAME], [$PMILIBNAME])
AC_SUBST([MPILIBNAME], [$MPILIBNAME])


AS_IF([test "x$enable_ncclomb" = xyes], [
       AC_CHECK_HEADERS([nccl.h], [],
                      [AC_MSG_ERROR([cannot include nccl.h])])
       AC_SEARCH_LIBS([ncclGetUniqueId], [nccl], [],
                      [AC_MSG_ERROR([cannot link with -lnccl])])
       AC_CHECK_FUNC([ncclSend], [nccl_pt2pt=true],
                     [AC_MSG_WARN([NCCL version >= 2.7 required for point-to-point support. Disabling building NCCL point-to-point benchmarks.])])
       AC_DEFINE([_ENABLE_NCCL_], [1], [Enable NCCL])
      ])

AS_IF([test "x$enable_openacc" = xyes], [
       AC_CHECK_HEADERS([openacc.h], [],
                        [AC_MSG_ERROR([cannot include openacc.h])])
       AC_DEFINE([_ENABLE_OPENACC_], [1], [Enable OpenACC])
       ])

AS_IF([test "x$enable_rocm" = xyes], [
       AC_DEFINE([__HIP_PLATFORM_HCC__])
       AC_CHECK_HEADERS([hip/hip_runtime_api.h], [],
                        [AC_MSG_ERROR([cannot include hip/hip_runtime_api.h])])
       AC_SEARCH_LIBS([hipFree], [amdhip64], [],
                      [AC_MSG_ERROR([cannot link with -lamdhip64])])
       AC_DEFINE([_ENABLE_ROCM_], [1], [Enable ROCm])
       ])

AS_CASE([$enable_cuda],
        [yes], [build_cuda_kernels=yes; build_cuda=yes],
        [basic], [build_cuda=yes])

AS_IF([test "x$build_cuda" = xyes], [
       AC_SEARCH_LIBS([cuPointerGetAttribute], [cuda], [],
                      [AC_MSG_ERROR([cannot link with -lcuda])])
       AC_SEARCH_LIBS([cudaFree], [cudart], [],
                      [AC_MSG_ERROR([cannot link with -lcudart])])
       AC_CHECK_HEADERS([cuda.h], [],
                        [AC_MSG_ERROR([cannot include cuda.h])])
       AC_DEFINE([_ENABLE_CUDA_], [1], [Enable CUDA])
       ])

AS_IF([test "x$build_cuda_kernels" = xyes], [
       AC_DEFINE([_ENABLE_CUDA_KERNEL_], [1], [Enable CUDA Kernel])
       ])

AS_IF([test "x$enable_nextsilicon" = xyes], [
       AC_DEFINE([_ENABLE_NEXTSILICON_], [1], [Enable NextSilicon support])
       ])

AS_CASE([$enable_papi],
        [yes], [build_papi=yes])

AS_IF([test "x$build_papi" = xyes], [
       AC_SEARCH_LIBS([PAPI_library_init], [papi], [],
                      [AC_MSG_ERROR([cannot link with -lpapi. Check provided
                       PAPI installation path in --with-papi])])
       AC_CHECK_HEADERS([papi.h], [],
                        [AC_MSG_ERROR([cannot include papi.h. Check provided
                         PAPI installation path in --with-papi])])
       AC_DEFINE([_ENABLE_PAPI_], [1], [Enable PAPI])
       ])


AS_IF([test "x$oshm_13_library" = xtrue], [
       AC_DEFINE([OSHM_1_3], [1], [Enable OpenSHMEM 1.3 features])
       ])
AM_CONDITIONAL([MPI2_LIBRARY], [test x$mpi2_library = xtrue])
AM_CONDITIONAL([MPI3_LIBRARY], [test x$mpi3_library = xtrue])
AM_CONDITIONAL([C], [test x$enable_c != xno])
AM_CONDITIONAL([CUDA], [test x$build_cuda = xyes])
AM_CONDITIONAL([NCCL], [test x$enable_ncclomb = xyes])
AM_CONDITIONAL([NCCL_PT2PT], [test x$nccl_pt2pt = xtrue])
AM_CONDITIONAL([CUDA_KERNELS], [test x$build_cuda_kernels = xyes])
AM_CONDITIONAL([OPENACC], [test x$enable_openacc = xyes])
AM_CONDITIONAL([ROCM], [test x$enable_rocm = xyes])
AM_CONDITIONAL([OSHM], [test x$oshm_library = xtrue])
AM_CONDITIONAL([MPI], [test x$mpi_library = xtrue])
AM_CONDITIONAL([UCX], [test x$with_ucx = xyes])
AM_CONDITIONAL([UPC], [test x$upc_compiler = xtrue])
AM_CONDITIONAL([UPCXX], [test x$upcxx_compiler = xtrue])
AM_CONDITIONAL([PAPI], [test x$enable_papi = xyes])
AM_CONDITIONAL([BUILD_USE_PGI], [`$CXX -V 2>&1 | grep pgc++ > /dev/null 2>&1`])

AC_DEFINE([FIELD_WIDTH], [18], [Width of field used to report numbers])
AC_DEFINE([FLOAT_PRECISION], [2], [Precision of reported numbers])

AC_CONFIG_FILES([Makefile c/Makefile c/mpi/Makefile c/mpi/pt2pt/Makefile
                 c/mpi/startup/Makefile c/mpi/one-sided/Makefile
                 c/mpi/collective/Makefile c/openshmem/Makefile
                 c/upc/Makefile c/upcxx/Makefile c/nccl/Makefile
                 c/nccl/collective/Makefile c/nccl/pt2pt/Makefile
               ])

AC_OUTPUT
