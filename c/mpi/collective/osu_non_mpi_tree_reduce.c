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
#include <osu_util_mpi.h>

int tree_reduce(int rank, const int np, void *sendbuf, void *recvbuf,
                int cnt, MPI_Datatype datatype, MPI_Comm comm)
{
    int num = np;
    int isodd = 0;
    int depth = 1;
    // many sends during a reduce process
    while (num > 1) {
        if (rank < num) {
            isodd = num % 2;
            // odd send to (odd - 1)
            if (rank % 2 != 0) {
                MPI_Send(sendbuf, cnt, datatype, (rank - 1) * depth, 0, comm);
                return 0;
            }

            if (rank != (num - 1)) {
                MPI_Recv(recvbuf, cnt, datatype, (rank + 1) * depth, 0, comm, MPI_STATUS_IGNORE);
                int_sum(sendbuf, recvbuf, cnt);
            }

            rank  /= 2;
            depth *= 2;
        }
        num = num / 2 + isodd;
    }
    if (rank == 0) {
        memcpy(recvbuf, sendbuf, cnt * sizeof(int));
    }
    return 0;
}

int main(int argc, char *argv[])
{
    int i, j, numprocs, rank, size;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double timer=0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0;
    int *sendbuf, *recvbuf;
    int po_ret;
    int errors = 0, local_errors = 0;
    size_t bufsize;
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

    for (size = options.min_message_size; size * sizeof(int) <=
            options.max_message_size; size *= 2) {
        if (size > LARGE_MESSAGE_SIZE) {
            options.skip = options.skip_large;
            options.iterations = options.iterations_large;
        }

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
                    MPI_CHECK(tree_reduce(rank, numprocs, sendbuf, recvbuf, size,
                                          MPI_INT, MPI_COMM_WORLD));
                    // MPI_CHECK(MPI_Reduce(sendbuf, recvbuf, size, MPI_INT,
                    //           MPI_SUM, 0, MPI_COMM_WORLD ));
                }
                if (0 == rank) {
                    set_buffer_validation(sendbuf, recvbuf, size, options.accel,
                            i);
                } else {
                    set_buffer_validation(sendbuf, NULL, size, options.accel,
                            i);
                }
            }
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

            apply_imbalance(options.imbalance,
                            options.imbalance_expected,
                            options.imbalance_variance);

            t_start = MPI_Wtime();


            MPI_CHECK(tree_reduce(rank, numprocs, sendbuf, recvbuf, size, MPI_INT,
                                  MPI_COMM_WORLD));
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
