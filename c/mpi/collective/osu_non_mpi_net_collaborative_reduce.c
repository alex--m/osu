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
#include <sys/shm.h>
#include <assert.h>

struct collaborative_slot {
    int counter;
    int data[];
};

static inline MPI_Aint collaborative_get_slot_disp(MPI_Aint base, int cnt, int idx)
{
    return base + ((cnt + 1) * idx);
}

int collaborative_reduce(MPI_Win win, volatile int *all_slots, const int rank,
                         const int np, void *sendbuf, void *recvbuf, int cnt,
                         MPI_Group group, MPI_Comm comm)
{
    int i, n;
    int next_slot_counter;
    volatile struct collaborative_slot *sum = (volatile struct collaborative_slot *)all_slots;
    MPI_Aint my_slot_disp, next_slot_disp;

    if (rank == 0) {
        assert(sum->counter == 0);
        assert(sum->data[0] == 0);
    }

    my_slot_disp   = collaborative_get_slot_disp(0, cnt, rank);
    next_slot_disp = collaborative_get_slot_disp(my_slot_disp, cnt, 1);
    memcpy(&sum->data[0], sendbuf, cnt * sizeof(int));

retry:
    if (rank) {
        MPI_Win_start(group, 0, win);
        MPI_CHECK(MPI_Get(&next_slot_counter, 1, MPI_INT, 0, next_slot_disp, 1, MPI_INT, win));
    } else {
        MPI_Win_wait(win);
        next_slot_counter = all_slots[next_slot_disp];
    }

    while (next_slot_counter) {
        if (rank) {
            MPI_CHECK(MPI_Get(recvbuf, cnt, MPI_INT, 0, next_slot_disp + 1, cnt, MPI_INT, win));
            int_sum(&sum->data[0], recvbuf, cnt);
        } else {
            int_sum(&sum->data[0], all_slots + next_slot_disp + 1, cnt);
        }

        next_slot_disp = collaborative_get_slot_disp(next_slot_disp, cnt, next_slot_counter);

        if (rank) {
            MPI_CHECK(MPI_Get(&next_slot_counter, 1, MPI_INT, 0, next_slot_disp, 1, MPI_INT, win));
        } else {
            next_slot_counter = all_slots[next_slot_disp];
        }
    }


    if (rank == 0) {
        if (next_slot_disp < collaborative_get_slot_disp(0, cnt, np)) {
            goto retry;
        }
        if (next_slot_disp != collaborative_get_slot_disp(0, cnt, np))
printf("ASSERT? next_slot_disp=%i collaborative_get_slot_disp(0, cnt, np)=%i\n\n", next_slot_disp, collaborative_get_slot_disp(0, cnt, np));

        assert(next_slot_disp == collaborative_get_slot_disp(0, cnt, np));
        memcpy(recvbuf, &sum->data[0], cnt * sizeof(int));
	MPI_Win_post(group, 0, win);
    } else {
        sum->counter = (next_slot_disp - my_slot_disp) / (cnt + 1);
        MPI_CHECK(MPI_Put(&sum->data, cnt, MPI_INT, 0, my_slot_disp + 1, cnt, MPI_INT, win));
        MPI_CHECK(MPI_Put(&sum->counter, 1, MPI_INT, 0, my_slot_disp, 1, MPI_INT, win));
        MPI_Win_complete(win);
    }

    return 0;
}

int main(int argc, char *argv[])
{
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
    int *all_slots;
    MPI_Group group;
    MPI_Group group2;
    MPI_Win win;

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

        slots_size = (numprocs + 1) * (size + 1) * sizeof(int);
        MPI_Win_allocate((MPI_Aint)slots_size, /* size in bytes */
                         sizeof(int), /* displacement units */
                         MPI_INFO_NULL, /* info object */
                         MPI_COMM_WORLD, /* communicator */
                         &all_slots,
                         &win /* window object */);

        MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &group));
        if (rank) {
            MPI_CHECK(MPI_Group_incl(group, 1, &errors /* 0 */, &group2));
        } else {
            int rank_range[1][3] = {1, numprocs-1, 1};
            MPI_CHECK(MPI_Group_range_incl(group, 1, rank_range, &group2));
            MPI_Win_post(group2, 0, win);
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
                    MPI_CHECK(collaborative_reduce(win, all_slots, rank, numprocs,
                                                   sendbuf, recvbuf, size,
                                                   group2, MPI_COMM_WORLD));
                    // MPI_CHECK(MPI_Reduce(sendbuf, recvbuf, size, MPI_INT,
                    //           MPI_SUM, 0, MPI_COMM_WORLD ));
                    if (rank == 0) {
                        memset(all_slots, 0, slots_size);
                    }
                }
            }
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

            apply_imbalance(options.imbalance,
                            options.imbalance_expected,
                            options.imbalance_variance);

            t_start = MPI_Wtime();


            MPI_CHECK(collaborative_reduce(win, all_slots, rank, numprocs, sendbuf,
                                           recvbuf, size, group2, MPI_COMM_WORLD));
            // MPI_CHECK(MPI_Reduce(sendbuf, recvbuf, size, MPI_INT, MPI_SUM, 0,
            //           MPI_COMM_WORLD ));
            t_stop=MPI_Wtime();

            if (0 == rank) {
                if (options.validate) {
                    local_errors += validate_data(recvbuf, size, numprocs,
                            options.accel, i);
                }

                memset(all_slots, 0, slots_size);
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
        MPI_CHECK(MPI_Win_free(&win));
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

    MPI_CHECK(MPI_Group_free(&group2));
    MPI_CHECK(MPI_Group_free(&group));
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
