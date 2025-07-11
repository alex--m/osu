AUTOMAKE_OPTIONS = subdir-objects

NVCC = nvcc
NVCFLAGS = -cuda -maxrregcount 32 -ccbin $(CXX) $(NVCCFLAGS)
SUFFIXES = .cu .cpp
.cu.cpp:
	$(NVCC) $(NVCFLAGS) $(INCLUDES) $(CPPFLAGS) --output-file $@.ii $<
	mv $@.ii $@

collectivedir = $(pkglibexecdir)/mpi/collective
collective_PROGRAMS = osu_alltoallv osu_alltoallw osu_allgatherv osu_scatterv \
					  osu_gatherv osu_reduce_scatter osu_barrier osu_reduce \
					  osu_reduce_integrity osu_reduce_dv osu_allreduce \
					  osu_allreduce_integrity osu_allreduce_dv osu_alltoall \
					  osu_bcast osu_bcast_integrity osu_bcast_dv osu_bcast_dv_rv \
					  osu_gather osu_allgather osu_scatter osu_iallgather osu_ibcast \
					  osu_ialltoall osu_ibarrier osu_igather osu_iscatter \
					  osu_iscatterv osu_igatherv osu_iallgatherv \
					  osu_ialltoallv osu_ialltoallw osu_ireduce osu_iallreduce \
					  osu_ireduce_scatter osu_non_mpi_tree_reduce \
					  osu_non_mpi_shm_collaborative_reduce osu_openmp_reduce \
					  osu_non_mpi_net_collaborative_reduce

AM_CFLAGS = -I${top_srcdir}/c/util
AM_CXXFLAGS = -I${top_srcdir}/c/util
AM_LDFLAGS = -fopenmp

UTILITIES = ../../util/osu_util.c ../../util/osu_util.h \
../../util/osu_util_mpi.c ../../util/osu_util_mpi.h \
../../util/osu_util_graph.c ../../util/osu_util_graph.h \
../../util/osu_util_papi.c ../../util/osu_util_papi.h
if CUDA_KERNELS
UTILITIES += ../../util/kernel.cu
if BUILD_USE_PGI
AM_CXXFLAGS = --nvcchost --no_preincludes
endif
endif

osu_allgatherv_SOURCES = osu_allgatherv.c $(UTILITIES)
osu_scatterv_SOURCES = osu_scatterv.c $(UTILITIES)
osu_gather_SOURCES = osu_gather.c $(UTILITIES)
osu_gatherv_SOURCES = osu_gatherv.c $(UTILITIES)
osu_reduce_scatter_SOURCES = osu_reduce_scatter.c $(UTILITIES)
osu_barrier_SOURCES = osu_barrier.c $(UTILITIES)
osu_reduce_SOURCES = osu_reduce.c $(UTILITIES)
osu_reduce_integrity_SOURCES = osu_reduce_integrity.c $(UTILITIES)
osu_reduce_dv_SOURCES = osu_reduce_dv.c $(UTILITIES)
osu_allreduce_SOURCES = osu_allreduce.c $(UTILITIES)
osu_allreduce_integrity_SOURCES = osu_allreduce_integrity.c $(UTILITIES)
osu_allreduce_dv_SOURCES = osu_allreduce_dv.c $(UTILITIES)
osu_bcast_SOURCES = osu_bcast.c $(UTILITIES)
osu_bcast_integrity_SOURCES = osu_bcast_integrity.c $(UTILITIES)
osu_bcast_dv_SOURCES = osu_bcast_dv.c $(UTILITIES)
osu_bcast_dv_rv_SOURCES = osu_bcast_dv_rv.c $(UTILITIES)
osu_alltoall_SOURCES = osu_alltoall.c $(UTILITIES)
osu_alltoallv_SOURCES = osu_alltoallv.c $(UTILITIES)
osu_alltoallw_SOURCES = osu_alltoallw.c $(UTILITIES)
osu_ialltoall_SOURCES = osu_ialltoall.c $(UTILITIES)
osu_ialltoallv_SOURCES = osu_ialltoallv.c $(UTILITIES)
osu_ialltoallw_SOURCES = osu_ialltoallw.c $(UTILITIES)
osu_ibarrier_SOURCES = osu_ibarrier.c $(UTILITIES)
osu_ibcast_SOURCES = osu_ibcast.c $(UTILITIES)
osu_igather_SOURCES = osu_igather.c $(UTILITIES)
osu_igatherv_SOURCES = osu_igatherv.c $(UTILITIES)
osu_allgather_SOURCES = osu_allgather.c $(UTILITIES)
osu_iallgather_SOURCES = osu_iallgather.c $(UTILITIES)
osu_iallgatherv_SOURCES = osu_iallgatherv.c $(UTILITIES)
osu_scatter_SOURCES = osu_scatter.c $(UTILITIES)
osu_iscatter_SOURCES = osu_iscatter.c $(UTILITIES)
osu_iscatterv_SOURCES = osu_iscatterv.c $(UTILITIES)
osu_ireduce_SOURCES = osu_ireduce.c $(UTILITIES)
osu_iallreduce_SOURCES = osu_iallreduce.c $(UTILITIES)
osu_ireduce_scatter_SOURCES = osu_ireduce_scatter.c $(UTILITIES)
osu_non_mpi_tree_reduce_SOURCES = osu_non_mpi_tree_reduce.c $(UTILITIES)
osu_non_mpi_shm_collaborative_reduce_SOURCES = osu_non_mpi_shm_collaborative_reduce.c $(UTILITIES)
osu_non_mpi_net_collaborative_reduce_SOURCES = osu_non_mpi_net_collaborative_reduce.c $(UTILITIES)
osu_openmp_reduce_SOURCES = osu_openmp_reduce.cpp $(UTILITIES)

if EMBEDDED_BUILD
    AM_CPPFLAGS = -fopenmp \
                  -I$(top_builddir)/../src/include \
                  -I${top_srcdir}/util \
                  -I${top_srcdir}/../src/include
if BUILD_PROFILING_LIB
    AM_LDFLAGS += $(top_builddir)/../lib/lib@PMPILIBNAME@.la
endif
    AM_LDFLAGS += $(top_builddir)/../lib/lib@MPILIBNAME@.la
else
    AM_CPPFLAGS = -fopenmp
endif

if OPENACC
    AM_CFLAGS += -acc -ta=tesla:nordc
    AM_CXXFLAGS = -acc -ta=tesla:nordc
endif
