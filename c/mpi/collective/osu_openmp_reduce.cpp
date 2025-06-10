#define BENCHMARK "OSU MPI%s Reduce Latency Test"
/*
 * Copyright (C) 2002-2022 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

extern "C" {
#include <osu_util_mpi.h>
}

#include <sys/shm.h>
#include <algorithm>
#include <assert.h>
#include <numeric>
#include <vector>
#include <omp.h>

#pragma omp declare reduction(vec_int_plus : std::vector<int> : \
                              std::transform(omp_out.begin(), omp_out.end(), \
                                             omp_in.begin(), omp_out.begin(), \
                                             std::plus<int>())) \
                    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

std::vector<int>& operator+=(std::vector<int> &lhs, const std::vector<int> &rhs) {
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), std::plus<int>());
    return lhs;
}

int openmp_reduce(int *all_slots, const int rank, const int np, void *sendbuf,
                  void *recvbuf, int cnt, MPI_Comm comm)
{
    std::vector<int> res(all_slots, all_slots + cnt);

    if (rank != 0) {
        return 0;
    }

    #pragma omp parallel for reduction(vec_int_plus : res)
    for (int i = 1; i < np; i++) {
        volatile int *my_slot =  (int*)all_slots + cnt * i;
        res += std::vector<int>(my_slot, my_slot + cnt);
    }

    memcpy(recvbuf, res.data(), cnt * sizeof(int));

    return 0;
}

int main(int argc, char *argv[])
{
    int *all_slots, *my_slot;
    int i, j, numprocs, rank, size, shmid;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double timer=0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0;
    int *sendbuf, *recvbuf;
    int po_ret;
    int errors = 0, local_errors = 0;
    size_t bufsize, slots_size;
    omb_graph_options_t omb_graph_options;
    omb_graph_data_t *omb_graph_data = NULL;
    int papi_eventset = OMB_PAPI_NULL;

    set_header(HEADER);
    set_benchmark_name("osu_reduce");

    options.bench = COLLECTIVE;
    options.subtype = REDUCE;

    po_ret = process_options(argc, argv);

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));

    omb_graph_options_init(&omb_graph_options);
    switch (po_ret) {
        case PO_BAD_USAGE:
            print_bad_usage_message(rank);
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        case PO_HELP_MESSAGE:
            print_help_message(rank);
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_VERSION_MESSAGE:
            print_version_message(rank);
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }

    if (numprocs < 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires at least two processes\n");
        }

        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }
    check_mem_limit(numprocs);
    options.min_message_size /= sizeof(int);
    if (options.min_message_size < MIN_MESSAGE_SIZE) {
        options.min_message_size = MIN_MESSAGE_SIZE;
    }

    bufsize = sizeof(int) * (options.max_message_size / sizeof(int));
    if (allocate_memory_coll((void**)&recvbuf, bufsize,
                options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    }
    set_buffer(recvbuf, options.accel, 1, bufsize);

    bufsize = sizeof(int) * (options.max_message_size / sizeof(int));
    if (allocate_memory_coll((void**)&sendbuf, bufsize,
                options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    }
    set_buffer(sendbuf, options.accel, 0, bufsize);

    print_preamble(rank);
    omb_papi_init(&papi_eventset);
    omp_set_num_threads(numprocs);

    for (size = options.min_message_size; size * sizeof(int) <=
            options.max_message_size; size *= 2) {
        if (size > LARGE_MESSAGE_SIZE) {
            options.skip = options.skip_large;
            options.iterations = options.iterations_large;
        }

        if (rank == 0) {
            slots_size = (numprocs + 1) * size * sizeof(int);
            shmid = shmget(IPC_PRIVATE, slots_size, IPC_CREAT | 0600);
        }
        MPI_Bcast(&shmid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        all_slots = (int*)shmat(shmid, NULL, 0);
        if (all_slots == (void *) -1)
            return -1;
        my_slot = all_slots + size * rank;

        omb_graph_allocate_and_get_data_buffer(&omb_graph_data,
                &omb_graph_options, size * sizeof(int), options.iterations);
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        timer=0.0;

        for (i = 0; i < options.iterations + options.skip; i++) {
            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
            }
            if (options.validate) {
                if (0 == rank) {
                    set_buffer_validation(sendbuf, recvbuf, size, options.accel,
                            i);
                } else {
                    set_buffer_validation(sendbuf, NULL, size, options.accel,
                            i);
                }
                for (j = 0; j < options.warmup_validation; j++) {
                    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                    memcpy(my_slot, sendbuf, size * sizeof(int));
                    MPI_CHECK(openmp_reduce(all_slots, rank, numprocs, sendbuf,
                                            recvbuf, size, MPI_COMM_WORLD));
                    // MPI_CHECK(MPI_Reduce(sendbuf, recvbuf, size, MPI_INT,
                    //           MPI_SUM, 0, MPI_COMM_WORLD ));
                }
            }
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

            apply_imbalance(options.imbalance,
                            options.imbalance_expected,
                            options.imbalance_variance);

            memcpy(my_slot, sendbuf, size * sizeof(int));

            t_start = MPI_Wtime();

            MPI_CHECK(openmp_reduce(all_slots, rank, numprocs, sendbuf, recvbuf,
                                    size, MPI_COMM_WORLD));
            // MPI_CHECK(MPI_Reduce(sendbuf, recvbuf, size, MPI_INT, MPI_SUM, 0,
            //           MPI_COMM_WORLD ));
            t_stop=MPI_Wtime();

            if (0 == rank) {
                if (options.validate) {
                    local_errors += validate_data(recvbuf, size, numprocs,
                            options.accel, i);
                }
            }

            if (i >= options.skip) {
                timer += t_stop - t_start;
                if (options.graph && 0 == rank) {
                    omb_graph_data->data[i - options.skip] = (t_stop -
                            t_start) * 1e6;
                }
            }
        }
        omb_papi_stop_and_print(&papi_eventset, size * sizeof(int));
        latency = (double)(timer * 1e6) / options.iterations;

        MPI_CHECK(MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0,
                MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD));
        avg_time = avg_time/numprocs;
        if (options.validate) {
            MPI_CHECK(MPI_Allreduce(&local_errors, &errors, 1, MPI_INT, MPI_SUM,
                        MPI_COMM_WORLD));
        }

        if (options.validate) {
            print_stats_validate(rank, size * sizeof(int), avg_time, min_time,
                    max_time, errors);
        } else {
            print_stats(rank, size * sizeof(int), avg_time, min_time,
                    max_time);
        }
        if (options.graph && 0 == rank) {
            omb_graph_data->avg = avg_time;
        }
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        if (0 != errors) {
            break;
        }
    }
    if (0 == rank && options.graph) {
        omb_graph_plot(&omb_graph_options, benchmark_name);
    }
    omb_graph_combined_plot(&omb_graph_options, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_options);
    omb_papi_free(&papi_eventset);

    free_buffer(recvbuf, options.accel);
    free_buffer(sendbuf, options.accel);

    MPI_CHECK(MPI_Finalize());

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }
    if (0 != errors && options.validate && 0 == rank ) {
        fprintf(stdout, "DATA VALIDATION ERROR: %s exited with status %d on"
                " message size %d.\n", argv[0], EXIT_FAILURE, size);
        exit(EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}
