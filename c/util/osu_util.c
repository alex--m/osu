/*
 * Copyright (C) 2002-2022 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level directory.
 */

#include "osu_util.h"

#ifdef _ENABLE_OPENACC_
#include <openacc.h>
#endif

/*
 * GLOBAL VARIABLES
 */
char const * benchmark_header = NULL;
char const * benchmark_name = NULL;
int accel_enabled = 0;
struct options_t options;

struct bad_usage_t bad_usage;

void print_header(int rank, int full)
{
    switch(options.bench) {
        case MBW_MR :
        case PT2PT :
            if (0 == rank) {
                if (options.omb_enable_ddt) {
                    fprintf(stdout, "# Set Derived DataTypes block_length to"
                            " %zu, stride to %zu\n",
                            options.ddt_type_parameters.block_length,
                            options.ddt_type_parameters.stride);
                }
                switch (options.accel) {
                    case CUDA:
                        printf(benchmark_header, "-CUDA");
                        break;
                    case OPENACC:
                        printf(benchmark_header, "-OPENACC");
                        break;
                    case ROCM:
                        printf(benchmark_header, "-ROCM");
                        break;
                    default:
                        printf(benchmark_header, "");
                        break;
                }

                switch (options.accel) {
                    case CUDA:
                    case OPENACC:
                    case ROCM:
                        fprintf(stdout, "# Send Buffer on %s and Receive Buffer on %s\n",
                                'M' == options.src ? ('D' == options.MMsrc ? "MANAGED (MD)" : "MANAGED (MH)") :
                                ('D' == options.src ? "DEVICE (D)" : "HOST (H)"),
                                'M' == options.dst ? ('D' == options.MMdst ? "MANAGED (MD)" : "MANAGED (MH)"):
                                ('D' == options.dst ? "DEVICE (D)" : "HOST (H)"));
                    default:
                        if (options.subtype == BW && options.bench != MBW_MR) {
                            fprintf(stdout, "%-*s%*s", 10, "# Size", FIELD_WIDTH, "Bandwidth (MB/s)");
                        } else if (options.subtype == LAT) {
                            fprintf(stdout, "%-*s%*s", 10, "# Size", FIELD_WIDTH, "Latency (us)");
                        } else if (options.subtype == LAT_MP) {
                            fprintf(stdout, "%-*s%*s", 10, "# Size", FIELD_WIDTH,"Latency (us)");
                        } else if (options.subtype == LAT_MT) {
                            fprintf(stdout, "%-*s%*s", 10, "# Size", FIELD_WIDTH, "Latency (us)");
                        }
                        if (options.validate && !(options.subtype == BW && options.bench == MBW_MR)) {
                            fprintf(stdout, "%*s", FIELD_WIDTH, "Validation");
                        }
                        if (options.omb_enable_ddt && !(options.subtype == BW &&
                                    options.bench == MBW_MR)) {
                            fprintf(stdout, "%*s", FIELD_WIDTH, "Transmit Size");
                        }
                        fprintf(stdout, "\n");
                        fflush(stdout);
                }
            }
            break;
        case COLLECTIVE :
            if (rank == 0) {
                fprintf(stdout, HEADER, "");

                if (options.show_size) {
                    fprintf(stdout, "%-*s", 10, "# Size");
                    fprintf(stdout, "%*s", FIELD_WIDTH, "Avg Latency(us)");
                }

                else {
                    fprintf(stdout, "# Avg Latency(us)");
                }

                if (full) {
                    fprintf(stdout, "%*s", FIELD_WIDTH, "Min Latency(us)");
                    fprintf(stdout, "%*s", FIELD_WIDTH, "Max Latency(us)");
                    fprintf(stdout, "%*s\n", 12, "Iterations");
                }

                else {
                    fprintf(stdout, "\n");
                }

                fflush(stdout);
            }
            break;
        default:
            break;
    }
}

void print_data (int rank, int full, int size, double avg_time,
                 double min_time, double max_time, int iterations)
{
    if (rank == 0) {
        if (options.show_size) {
            fprintf(stdout, "%-*d", 10, size);
            fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, avg_time);
        } else {
            fprintf(stdout, "%*.*f", 17, FLOAT_PRECISION, avg_time);
        }

        if (full) {
            fprintf(stdout, "%*.*f%*.*f%*d\n",
                    FIELD_WIDTH, FLOAT_PRECISION, min_time,
                    FIELD_WIDTH, FLOAT_PRECISION, max_time,
                    12, iterations);
        } else {
            fprintf(stdout, "\n");

        }

        fflush(stdout);
    }
}


static int set_min_message_size (long long value)
{
    if (0 >= value) {
        return -1;
    }

    options.min_message_size = value;

    return 0;
}

static int set_max_message_size (long long value)
{
    if (0 > value) {
        return -1;
    }

    options.max_message_size = value;

    return 0;
}

static int set_message_size (char *val_str)
{
    int retval = -1;
    int i, count = 0;
    char *val1, *val2;

    for (i=0; val_str[i]; i++) {
        if (val_str[i] == ':')
            count++;
    }

    if (!count) {
        retval = set_max_message_size(atoll(val_str));
    } else if (count == 1) {
        val1 = strtok(val_str, ":");
        val2 = strtok(NULL, ":");

        if (val1 && val2) {
            retval = set_min_message_size(atoll(val1));
            retval = set_max_message_size(atoll(val2));
        } else if (val1) {
            if (val_str[0] == ':') {
                retval = set_max_message_size(atoll(val1));
            } else {
                retval = set_min_message_size(atoll(val1));
            }
        }
    }

    return retval;
}

static int set_receiver_threads (int value)
{
    if (MIN_NUM_THREADS > value || value >= MAX_NUM_THREADS) {
        return -1;
    }

    options.num_threads = value;

    return 0;

}

static int set_sender_threads (int value)
{
    if (MIN_NUM_THREADS > value || value >= MAX_NUM_THREADS) {
        return -1;
    }

    options.sender_thread = value;

    return 0;

}

static int set_threads (char *val_str)
{
    int retval = -1;
    int i, count = 0;
    char *val1, *val2;

    for (i=0; val_str[i]; i++) {
        if (val_str[i] == ':')
            count++;
    }

    if (!count) {
        retval = set_receiver_threads(atoi(val_str));
        options.sender_thread = -1;
    } else if (count == 1) {
        val1 = strtok(val_str, ":");
        val2 = strtok(NULL, ":");

        if (val1 && val2) {
            retval = set_sender_threads(atoi(val1));
            if (retval == -1) {
                return retval;
            }
            retval = set_receiver_threads(atoi(val2));
            if (retval == -1) {
                return retval;
            }
        }

    }

    return retval;
}

static int set_receiver_processes (int value)
{
    if (MIN_NUM_PROCESSES > value || value >= MAX_NUM_PROCESSES) {
        return -1;
    }

    options.num_processes = value;

    return 0;

}

static int set_sender_processes (int value)
{
    if (MIN_NUM_PROCESSES > value || value >= MAX_NUM_PROCESSES) {
        return -1;
    }

    options.sender_processes = value;

    return 0;

}

static int set_processes (char *val_str)
{
    int retval = -1;
    int i, count = 0;
    char *val1, *val2;

    for (i=0; val_str[i]; i++) {
        if (val_str[i] == ':')
            count++;
    }

    if (!count) {
        retval = set_receiver_processes(atoi(val_str));
        options.sender_processes = -1;
    } else if (count == 1) {
        val1 = strtok(val_str, ":");
        val2 = strtok(NULL, ":");

        if (val1 && val2) {
            retval = set_sender_processes(atoi(val1));
            if (retval == -1) {
                return retval;
            }
            retval = set_receiver_processes(atoi(val2));
            if (retval == -1) {
                return retval;
            }
        }

    }

    return retval;
}

static int set_num_warmup (int value)
{
    if (0 > value) {
        return -1;
    }

    options.skip = value;
    options.skip_large = value;

    return 0;
}

static int set_num_warmup_validation (int value)
{
    if (0 > value) {
        return -1;
    }

    options.warmup_validation = value;

    return 0;
}

static int set_num_iterations (int value)
{
    if (1 > value) {
        return -1;
    }

    options.iterations = value;
    options.iterations_large = value;

    return 0;
}

static int set_window_size (int value)
{
    if (1 > value) {
        return -1;
    }

    options.window_size = value;

    return 0;
}

static int set_device_array_size (int value)
{
    if (value < 1 ) {
        return -1;
    }

    options.device_array_size = value;

    return 0;
}

static int set_num_probes (int value)
{
    if (value < 0 ) {
        return -1;
    }

    options.num_probes = value;

    return 0;
}

static int set_max_memlimit (long long value)
{
    options.max_mem_limit = value;

    if (value < MAX_MEM_LOWER_LIMIT) {
        options.max_mem_limit = MAX_MEM_LOWER_LIMIT;
        fprintf(stderr,"Requested memory limit too low, using [%d] instead.",
                MAX_MEM_LOWER_LIMIT);
    }

    return 0;
}

void set_header (const char * header)
{
    benchmark_header = header;
}

void set_benchmark_name (const char * name)
{
    benchmark_name = name;
}

void enable_accel_support (void)
{
    accel_enabled = ((CUDA_ENABLED || OPENACC_ENABLED || ROCM_ENABLED) &&
            !(options.subtype == LAT_MT || options.subtype == LAT_MP));
}

int process_options (int argc, char *argv[])
{
    extern char * optarg;
    extern int optind, optopt;

    char const * optstring = NULL;
    int c, ret = PO_OKAY;

    int option_index = 0;
    char *graph_term_type = NULL;
    int omb_long_options_itr = OMB_LONG_OPTIONS_ARRAY_SIZE - 1;
    static struct option long_options[OMB_LONG_OPTIONS_ARRAY_SIZE] = {
            {"help",                no_argument,        0,  'h'},
            {"version",             no_argument,        0,  'v'},
            {"full",                no_argument,        0,  'f'},
            {"message-size",        required_argument,  0,  'm'},
            {"window-size",         required_argument,  0,  'W'},
            {"num-test-calls",      required_argument,  0,  't'},
            {"iterations",          required_argument,  0,  'i'},
            {"warmup",              required_argument,  0,  'x'},
            {"imbalance",           required_argument,  0,  'I'},
            {"array-size",          required_argument,  0,  'a'},
            {"sync-option",         required_argument,  0,  's'},
            {"win-options",         required_argument,  0,  'w'},
            {"mem-limit",           required_argument,  0,  'M'},
            {"accelerator",         required_argument,  0,  'd'},
            {"cuda-target",         required_argument,  0,  'r'},
            {"print-rate",          required_argument,  0,  'R'},
            {"num-pairs",           required_argument,  0,  'p'},
            {"vary-window",         required_argument,  0,  'V'},
            {"validation",          no_argument,        0,  'c'},
            {"buffer-num",          required_argument,  0,  'b'},
            {"validation-warmup",   required_argument,  0,  'u'},
            {"graph",               required_argument,  0,  'G'},
            {"papi",                required_argument,  0,  'P'}
    };

    enable_accel_support();

    if (options.bench == PT2PT) {
        if (accel_enabled) {
            if (options.subtype == BW) {
                optstring = "+:x:i:t:m:d:W:hvb:cu:G:D:";
            } else {
                optstring = "+:x:i:m:d:hvcu:G:D:";
            }
        } else{
            if (options.subtype == LAT_MT) {
                optstring = "+:hvm:x:i:t:cu:G:D:";
            } else if (options.subtype == LAT_MP) {
                optstring = "+:hvm:x:i:t:cu:G:D:P:";
            } else if (options.subtype == BW) {
                optstring = "+:hvm:x:i:t:W:b:cu:G:D:P:";
            } else {
                optstring = "+:hvm:x:i:b:cu:G:D:P:";
            }
        }
        long_options[omb_long_options_itr].name = "ddt";
        long_options[omb_long_options_itr].has_arg = required_argument;
        long_options[omb_long_options_itr].flag = 0;
        long_options[omb_long_options_itr].val = 'D';
    } else if (options.bench == COLLECTIVE) {
        if (options.subtype == LAT ||
                options.subtype == BARRIER ||
                options.subtype == ALLTOALL ||
                options.subtype == GATHER ||
                options.subtype == REDUCE ||
                options.subtype == SCATTER ||
                options.subtype == REDUCE_SCATTER ||
                options.subtype == BCAST) { /* Blocking */
            if (options.subtype == GATHER ||
                    options.subtype == SCATTER ||
                    options.subtype == ALLTOALL ||
                    options.subtype == BCAST ) {
                optstring = "+:hvfm:i:x:I:M:a:cu:G:D:P:";
                if (accel_enabled) {
                    optstring = (CUDA_KERNEL_ENABLED) ?
                        "+:d:hvfm:i:x:I:M:r:a:cu:G:D:" :
                        "+:d:hvfm:i:x:I:M:a:cu:G:D:";
                }
                long_options[omb_long_options_itr].name = "ddt";
                long_options[omb_long_options_itr].has_arg = required_argument;
                long_options[omb_long_options_itr].flag = 0;
                long_options[omb_long_options_itr].val = 'D';
            } else {
                if (options.subtype == BARRIER) {
                    optstring = "+:hvfm:i:x:I:M:a:u:G:P:";
                    if (accel_enabled) {
                        optstring = (CUDA_KERNEL_ENABLED) ?
                            "+:d:hvfm:i:x:I:M:r:a:u:G:" :
                            "+:d:hvfm:i:x:I:M:a:u:G:";
                    }
                } else {
                    optstring = "+:hvfm:i:x:I:M:a:cu:G:P:";
                    if (accel_enabled) {
                        optstring = (CUDA_KERNEL_ENABLED) ?
                            "+:d:hvfm:i:x:I:M:r:a:cu:G:" :
                            "+:d:hvfm:i:x:I:M:a:cu:G:";
                    }
                }
            }
        } else if (options.subtype == NBC) {
            optstring = "+:hvfm:i:x:I:M:t:a:G:P:";
            if (accel_enabled) {
                optstring = (CUDA_KERNEL_ENABLED) ? "+:d:hvfm:i:x:I:M:t:r:a:G:" :
                    "+:d:hvfm:i:x:I:M:t:a:G:";
            }
        } else { /* Non-Blocking */
            if (options.subtype == NBC_GATHER ||
                    options.subtype == NBC_ALLTOALL ||
                    options.subtype == NBC_SCATTER ||
                    options.subtype == NBC_BCAST) {
                optstring = "+:hvfm:i:x:I:M:t:a:cu:G:D:P:";
                if (accel_enabled) {
                    optstring = (CUDA_KERNEL_ENABLED) ?
                        "+:d:hvfm:i:x:I:M:t:r:a:cu:G:D:" :
                        "+:d:hvfm:i:x:I:M:t:a:cu:G:D:";
                }
                long_options[omb_long_options_itr].name = "ddt";
                long_options[omb_long_options_itr].has_arg = required_argument;
                long_options[omb_long_options_itr].flag = 0;
                long_options[omb_long_options_itr].val = 'D';
            } else {
                optstring = "+:hvfm:i:x:I:M:t:a:cu:G:P:";
                if (accel_enabled) {
                    optstring = (CUDA_KERNEL_ENABLED) ? "+:d:hvfm:i:x:I:M:t:r:a:cu:G:"
                        : "+:d:hvfm:i:x:I:M:t:a:cu:G:";
                }
            }
        }
    } else if (options.bench == ONE_SIDED) {
        if(options.subtype == BW) {
            optstring = (accel_enabled) ? "+:w:s:hvm:d:x:i:W:G:" :
                "+:w:s:hvm:x:i:W:G:P:";
        } else {
            optstring = (accel_enabled) ? "+:w:s:hvm:d:x:i:G:" :
                "+:w:s:hvm:x:i:G:P:";
        }
    } else if (options.bench == MBW_MR) {
        optstring = (accel_enabled) ? "p:W:R:x:i:m:d:Vhvb:cu:G:D:" :
            "p:W:R:x:i:m:Vhvb:cu:G:D:P:";
        long_options[omb_long_options_itr].name = "ddt";
        long_options[omb_long_options_itr].has_arg = required_argument;
        long_options[omb_long_options_itr].flag = 0;
        long_options[omb_long_options_itr].val = 'D';
    } else if (options.bench == OSHM || options.bench == UPC || options.bench == UPCXX) {
        optstring = ":hvfm:i:M:";
    } else {
        fprintf(stderr,"Invalid benchmark type");
        exit(1);
    }

    /* Set default options*/
    options.accel = NONE;
    options.show_size = 1;
    options.show_full = 0;
    options.num_probes = 0;
    options.device_array_size = 32;
    options.target = CPU;
    options.min_message_size = MIN_MESSAGE_SIZE;
    if (options.bench == COLLECTIVE) {
        options.max_message_size = MAX_MSG_SIZE_COLL;
    } else {
        options.max_message_size = MAX_MESSAGE_SIZE;
    }
    options.max_mem_limit = MAX_MEM_LIMIT;
    options.window_size_large = WINDOW_SIZE_LARGE;
    options.window_size = WINDOW_SIZE_LARGE;
    options.window_varied = 0;
    options.print_rate = 1;
    options.validate = 0;
    options.papi_enabled = 0;
    options.imbalance = NO_IMBALANCE;
    options.buf_num = SINGLE;
    options.omb_enable_ddt = 0;
    options.ddt_type_parameters.block_length = OMB_DDT_BLOCK_LENGTH_DEFAULT;
    options.ddt_type_parameters.stride = OMB_DDT_STRIDE_DEFAULT;
    options.graph = 0;
    options.graph_output_term = 0;
    options.graph_output_png = 0;
    options.graph_output_pdf = 0;
    options.omb_enable_ddt = 0;
    options.ddt_type_parameters.block_length = OMB_DDT_BLOCK_LENGTH_DEFAULT;
    options.ddt_type_parameters.stride = OMB_DDT_STRIDE_DEFAULT;
    options.src = 'H';
    options.dst = 'H';

    switch (options.subtype) {
        case BW:
            options.iterations = BW_LOOP_SMALL;
            options.skip = BW_SKIP_SMALL;
            options.iterations_large = BW_LOOP_LARGE;
            options.skip_large = BW_SKIP_LARGE;
            options.warmup_validation = VALIDATION_SKIP_DEFAULT;
            break;
        case LAT_MT:
            options.num_threads = DEF_NUM_THREADS;
            options.min_message_size = 0;
            options.sender_thread=-1;
        case LAT_MP:
            options.num_processes = DEF_NUM_PROCESSES;
            options.min_message_size = 0;
            options.sender_processes = DEF_NUM_PROCESSES;
        case LAT:
        case BARRIER:
        case GATHER:
        case ALLTOALL:
        case NBC_ALLTOALL:
        case NBC_GATHER:
        case NBC_REDUCE:
        case NBC_SCATTER:
        case NBC_BCAST:
        case REDUCE:
        case SCATTER:
        case BCAST:
        case REDUCE_SCATTER:
        case NBC_REDUCE_SCATTER:
        case NBC:
            if (options.bench == COLLECTIVE) {
                options.iterations = COLL_LOOP_SMALL;
                options.skip = COLL_SKIP_SMALL;
                options.iterations_large = COLL_LOOP_LARGE;
                options.skip_large = COLL_SKIP_LARGE;
            } else {
                options.iterations = LAT_LOOP_SMALL;
                options.skip = LAT_SKIP_SMALL;
                options.iterations_large = LAT_LOOP_LARGE;
                options.skip_large = LAT_SKIP_LARGE;
            }
            if (options.bench == PT2PT) {
                options.min_message_size = 0;
            }
            options.warmup_validation = VALIDATION_SKIP_DEFAULT;
            break;
        default:
            break;
    }

    switch (options.bench) {
        case UPCXX:
        case UPC:
            options.show_size = 0;
        case OSHM:
            options.iterations = OSHM_LOOP_SMALL;
            options.skip = OSHM_SKIP_SMALL;
            options.iterations_large = OSHM_LOOP_LARGE;
            options.skip_large = OSHM_SKIP_LARGE;
            options.max_message_size = 1<<20;
            break;
        default:
            break;
    }

    while ((c = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1) {
        bad_usage.opt = c;
        bad_usage.optarg = NULL;
        bad_usage.message = NULL;

        switch(c) {
            case 'h':
                return PO_HELP_MESSAGE;
            case 'v':
                return PO_VERSION_MESSAGE;
            case 'm':
                if (set_message_size(optarg)) {
                    bad_usage.message = "Invalid Message Size";
                    bad_usage.optarg = optarg;

                    return PO_BAD_USAGE;
                }
                break;
            case 't':
                if (options.bench == COLLECTIVE) {
                    if (set_num_probes(atoi(optarg))) {
                        bad_usage.message = "Invalid Number of Probes";
                        bad_usage.optarg = optarg;

                        return PO_BAD_USAGE;
                    }
                } else if (options.bench == PT2PT) {
                    if (options.subtype == LAT_MT) {
                        if (set_threads(optarg)) {
                            bad_usage.message = "Invalid Number of Threads";
                            bad_usage.optarg = optarg;

                            return PO_BAD_USAGE;
                        }
                    } else if (options.subtype == LAT_MP) {
                        if (set_processes(optarg)) {
                            bad_usage.message = "Invalid Number of Processes";
                            bad_usage.optarg = optarg;

                            return PO_BAD_USAGE;
                        }
                    }
                }
                break;
            case 'i':
                if (set_num_iterations(atoi(optarg))) {
                    bad_usage.message = "Invalid Number of Iterations";
                    bad_usage.optarg = optarg;

                    return PO_BAD_USAGE;
                }
                break;
            case 'x':
                if (set_num_warmup(atoi(optarg))) {
                    bad_usage.message = "Invalid Number of Warmup Iterations";
                    bad_usage.optarg = optarg;

                    return PO_BAD_USAGE;
                }
                break;
            case 'I':
                if (PO_OKAY != set_imbalance(optarg)) {
                    bad_usage.message = "Invalid Imbalance Specification";
                    bad_usage.optarg = optarg;

                    return PO_BAD_USAGE;
                }
                break;
            case 'R':
                options.print_rate = atoi(optarg);
                if (0 != options.print_rate && 1 != options.print_rate) {
                    return PO_BAD_USAGE;
                }
                break;
            case 'W':
                if (set_window_size(atoi(optarg))) {
                    bad_usage.message = "Invalid Window Size";
                    bad_usage.optarg = optarg;

                    return PO_BAD_USAGE;
                }
                break;
            case 'V':
                options.window_varied = 1;
                break;
            case 'p':
                options.pairs = atoi(optarg);
                break;
            case 'a':
                if (set_device_array_size(atoi(optarg))) {
                    bad_usage.message = "Invalid Device Array Size";
                    bad_usage.optarg = optarg;

                    return PO_BAD_USAGE;
                }
                break;
            case 'f':
                options.show_full = 1;
                break;
            case 'M':
                /*
                 * This function does not error but prints a warning message if
                 * the value is too low.
                 */
                set_max_memlimit(atoll(optarg));
                break;
            case 'd':
                if (!accel_enabled) {
                    bad_usage.message = "Benchmark Does Not Support "
                            "Accelerator Transfers";
                    bad_usage.optarg = optarg;
                    return PO_BAD_USAGE;
                } else if (0 == strncasecmp(optarg, "cuda", 10)) {
                    if (CUDA_ENABLED) {
                        options.accel = CUDA;
                    } else {
                        bad_usage.message = "CUDA Support Not Enabled\n"
                                "Please recompile benchmark with CUDA support";
                        bad_usage.optarg = optarg;
                        return PO_BAD_USAGE;
                    }
                } else if (0 == strncasecmp(optarg, "managed", 10)) {
                    if (CUDA_ENABLED) {
                        options.accel = MANAGED;
                    } else {
                        bad_usage.message = "CUDA Managed Memory Support Not Enabled\n"
                                "Please recompile benchmark with CUDA support";
                        bad_usage.optarg = optarg;
                        return PO_BAD_USAGE;
                    }
                } else if (0 == strncasecmp(optarg, "openacc", 10)) {
                    if (OPENACC_ENABLED) {
                        options.accel = OPENACC;
                    } else {
                        bad_usage.message = "OpenACC Support Not Enabled\n"
                                "Please recompile benchmark with OpenACC support";
                        bad_usage.optarg = optarg;
                        return PO_BAD_USAGE;
                    }
                } else if (0 == strncasecmp(optarg, "rocm", 10)) {
                    if (ROCM_ENABLED) {
                        options.accel = ROCM;
                    } else {
                        bad_usage.message = "ROCm Support Not Enabled\n"
                                "Please recompile benchmark with ROCm support";
                        bad_usage.optarg = optarg;
                        return PO_BAD_USAGE;
                    }
                } else {
                    bad_usage.message = "Invalid Accel Type Specified";
                    bad_usage.optarg = optarg;
                    return PO_BAD_USAGE;
                }
                break;
            case 'r':
                if (CUDA_KERNEL_ENABLED) {
                    if (0 == strncasecmp(optarg, "cpu", 10)) {
                        options.target = CPU;
                    } else if (0 == strncasecmp(optarg, "gpu", 10)) {
                        options.target = GPU;
                    } else if (0 == strncasecmp(optarg, "both", 10)) {
                        options.target = BOTH;
                    } else {
                        bad_usage.message = "Please use cpu, gpu, or both";
                        bad_usage.optarg = optarg;
                        return PO_BAD_USAGE;
                    }
                } else {
                    bad_usage.message = "CUDA Kernel Support Not Enabled\n"
                            "Please recompile benchmark with CUDA Kernel support";
                    bad_usage.optarg = optarg;
                    return PO_BAD_USAGE;
                }
                break;
            case 'w':
                ret = process_one_sided_options(c, optarg);
                if (ret == PO_BAD_USAGE) {
                    bad_usage.message = "Invalid option or invalid argument";
                    bad_usage.opt = c;
                    bad_usage.optarg = optarg;
                    return PO_BAD_USAGE;
                }
                break;
            case 's':
                ret = process_one_sided_options(c, optarg);
                if (ret == PO_BAD_USAGE) {
                    bad_usage.message = "Invalid option or invalid argument";
                    bad_usage.opt = c;
                    bad_usage.optarg = optarg;
                    return PO_BAD_USAGE;
                }
                break;
            case 'c':
                options.validate = 1;
                if (options.omb_enable_ddt) {
                    bad_usage.message = "Derived data type does not support"
                        " validation";
                    bad_usage.optarg = optarg;
                    return PO_BAD_USAGE;
                }
                break;
            case 'P':
#ifdef _ENABLE_PAPI_
                options.papi_enabled = 1;
                omb_papi_parse_event_options(optarg);
#else
                bad_usage.message = "Invalid option. Please reconfigure with"
                    " PAPI.";
                bad_usage.opt = optopt;
                return PO_BAD_USAGE;
#endif
                break;
            case 'u':
                if (set_num_warmup_validation(atoi(optarg))) {
                    bad_usage.message = "Invalid Number of Validation Warmup "
                        " Iterations";
                    bad_usage.optarg = optarg;

                    return PO_BAD_USAGE;
                }
                if (options.warmup_validation >  VALIDATION_SKIP_MAX) {
                    bad_usage.message = "Number of Validation Warmup Iterations"
                        "must be less than 10";
                    bad_usage.optarg = optarg;

                    return PO_BAD_USAGE;
                }
                break;
            case 'b':
                if (0 == strncasecmp(optarg, "single", 10)) {
                    options.buf_num = SINGLE;
                } else if (0 == strncasecmp(optarg, "multiple", 10)) {
                    options.buf_num = MULTIPLE;
                } else {
                    bad_usage.message = "Please use 'single' or 'multiple' for buffer type";
                    bad_usage.optarg = optarg;
                    return PO_BAD_USAGE;
                }
                break;
            case 'G':
                options.graph = 1;
                graph_term_type = strtok(optarg, ",");
                if (NULL == graph_term_type) {
                    bad_usage.message = "Please pass graph"
                        " types[tty,png,pdf]\n";
                    bad_usage.optarg = optarg;
                    return PO_BAD_USAGE;
                }
                while(NULL != graph_term_type) {
                    if (0 == strncasecmp(graph_term_type, "png", 3)) {
                        options.graph_output_png = 1;
                    }
                    else if (0 == strncasecmp(graph_term_type, "tty", 3)) {
                        options.graph_output_term = 1;
                    }
                    else if (0 == strncasecmp(graph_term_type, "pdf", 3)) {
                        options.graph_output_pdf = 1;
                    } else {
                       bad_usage.message = "Invalid graph type. Valid graph"
                           " types[tty,png,pdf]\n";
                       bad_usage.optarg = optarg;
                       return PO_BAD_USAGE;
                    }
                    graph_term_type = strtok(NULL, ",");
                }
                break;
            case 'D':
                options.omb_enable_ddt = 1;
                if (options.validate) {
                    bad_usage.message = "Derived data type does not support"
                        " validation";
                    bad_usage.optarg = optarg;
                    return PO_BAD_USAGE;
                }
                ret = PO_OKAY;
                ret = omb_ddt_process_options(optarg, &bad_usage);
                if (ret == PO_BAD_USAGE) {
                    return ret;
                }
                break;
            case ':':
                bad_usage.message = "Option Missing Required Argument";
                bad_usage.opt = optopt;
                return PO_BAD_USAGE;
            default:
                bad_usage.message = "Invalid option";
                bad_usage.opt = optopt;
                return PO_BAD_USAGE;
        }
    }

    if (!options.validate) {
        options.warmup_validation = 0;
    }

    if (accel_enabled) {
        if ((optind + 2) == argc) {
            options.src = argv[optind][0];
            if (options.src == 'M')
            {
#ifdef _ENABLE_CUDA_KERNEL_
                options.MMsrc = argv[optind][1];
                if (options.MMsrc == '\0') {
                    fprintf(stderr, "The M flag for destination buffer is "
                            "deprecated. Please use MD or MH to set the "
                            "effective location of CUDA Unified Memory buffers "
                            "to be device or host respectively. Currently M "
                            "flag is considered as MH\n");
                    options.MMsrc = 'H';
                } else if (options.MMsrc != 'D' && options.MMsrc != 'H') {
                    fprintf(stderr, "Please use MD or MH to set the effective "
                            "location of CUDA Unified Memory buffers to be "
                            "device or host respectively\n");
                    return PO_BAD_USAGE;
                }
#else
                fprintf(stderr, "Managed memory support requires CUDA kernels.\n");
                return PO_BAD_USAGE;
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */
            }
            options.dst = argv[optind + 1][0];
            if (options.dst == 'M')
            {
#ifdef _ENABLE_CUDA_KERNEL_
                options.MMdst = argv[optind+1][1];
                if (options.MMdst == '\0') {
                    fprintf(stderr, "The M flag for destination buffer is "
                            "deprecated. Please use MD or MH to set the "
                            "effective location of CUDA Unified Memory buffers "
                            "to be device or host respectively. Currently M "
                            "flag is considered as MH\n");
                    options.MMdst = 'H';
                } else if (options.MMdst != 'D' && options.MMdst != 'H') {
                    fprintf(stderr, "Please use MD or MH to set the effective "
                            "location of CUDA Unified Memory buffers to be "
                            "device or host respectively\n");
                    return PO_BAD_USAGE;
                }
#else
                fprintf(stderr, "Managed memory support requires CUDA kernels.\n");
                return PO_BAD_USAGE;
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */
            }
            /* No need to check if '-d' is given */
            if (NONE == options.accel || options.bench == PT2PT) {
                setAccel(options.src);
                setAccel(options.dst);
            }
        } else if (optind != argc) {
            return PO_BAD_USAGE;
        }
    }

    return PO_OKAY;
}

int omb_ddt_process_options(char *optarg, struct bad_usage_t *bad_usage)
{
    char *option;
    if (NULL == optarg) {
        bad_usage->message = "Please pass a ddt"
            " type[cont,vect,indx]\n";
        bad_usage->optarg = optarg;
        return PO_BAD_USAGE;
    }
    option = strtok(optarg, ":");
    if (0 == strncasecmp(optarg, "vect", 4)) {
        options.ddt_type = OMB_DDT_VECTOR;
        option = strtok(NULL, ":");
        if (NULL != option) {
            options.ddt_type_parameters.stride = atoi(option);
        }
        option = strtok(NULL, ":");
        if (NULL != option) {
            options.ddt_type_parameters.block_length = atoi(option);
        }
    } else if (0 == strncasecmp(optarg, "indx", 4)) {
        options.ddt_type = OMB_DDT_INDEXED;
        option = strtok(NULL, ":");
        if (NULL != option) {
            if (OMB_DDT_FILE_PATH_MAX_LENGTH < strlen(option)) {
                fprintf(stderr, "ERROR: Max allowed size for filepath is:%d\n"
                        "To increase the max allowed filepath limit, update"
                        " OMB_DDT_FILE_PATH_MAX_LENGTH in c/util/osu_util.h.\n",
                        OMB_DDT_FILE_PATH_MAX_LENGTH);
                fflush(stderr);
                bad_usage->message = "Index DDT filepath exceeds maximum length"
                    " allowed";
                bad_usage->optarg = optarg;
                return PO_BAD_USAGE;

            }
            strcpy(options.ddt_type_parameters.filepath, option);
        }
    } else if (0 == strncasecmp(optarg, "cont", 4)) {
        options.ddt_type = OMB_DDT_CONTIGUOUS;
    } else {
        bad_usage->message = "Invalid ddt type. Valid ddt"
            " types[cont,vect,indx]\n";
        bad_usage->optarg = optarg;
        return PO_BAD_USAGE;
    }
    return PO_OKAY;
}

/* Set the initial accelerator type */
int setAccel(char buf_type)
{
    switch (buf_type) {
        case 'H':
            break;
        case 'M':
            /* For managed memory benchmarks, use multiple buffers to report
             * accurate performance numbers */
            options.buf_num = MULTIPLE;
        case 'D':
            if (options.bench != PT2PT && options.bench != ONE_SIDED && options.bench != MBW_MR) {
                bad_usage.opt = buf_type;
                bad_usage.message = "This argument is only supported for one-sided and pt2pt benchmarks";
                return PO_BAD_USAGE;
            }
            if (NONE == options.accel || MANAGED == options.accel) {
#if defined(_ENABLE_OPENACC_) && !defined(_ENABLE_CUDA_)
                options.accel = OPENACC;
#elif defined(_ENABLE_CUDA_)
                options.accel = CUDA;
#elif defined(_ENABLE_ROCM_)
                options.accel = ROCM;
#endif
            }
            break;
        default:
            return PO_BAD_USAGE;
    }
    return PO_OKAY;
}


double getMicrosecondTimeStamp()
{
    double retval;
    struct timeval tv;
    if (gettimeofday(&tv, NULL)) {
        perror("gettimeofday");
        abort();
    }
    retval = ((double)tv.tv_sec) * 1000000 + tv.tv_usec;
    return retval;
}

void wtime(double *t)
{
    static int sec = -1;
    struct timeval tv;
    //gettimeofday(&tv, (void *)0);
    gettimeofday(&tv, 0);
    if (sec < 0) sec = tv.tv_sec;
    *t = (tv.tv_sec - sec)*1.0e+6 + tv.tv_usec;
}

/* Set the initial accelerator type */
int set_imbalance(char *distribution_info)
{
    char *s = strtok(distribution_info, ":");
    if (!s) {
        return PO_BAD_USAGE;
    }

    switch (*s) {
        case 'U':
            options.imbalance          = UNIFORM;
            options.imbalance_expected = atol(strtok(NULL, ":"));
            options.imbalance_variance = 0;
            break;
        case 'G':
            options.imbalance          = GAUSSIAN;
            options.imbalance_expected = atol(strtok(NULL, ":"));
            options.imbalance_variance = atol(strtok(NULL, ":"));
            break;
        default:
            return PO_BAD_USAGE;
    }

    if ((options.imbalance_expected < 0) ||
        (options.imbalance_variance < 0)) {
        return PO_BAD_USAGE;
    }

    return PO_OKAY;
}

static inline long sample_uniform(unsigned expected_value)
{
    return ((double) rand() / (RAND_MAX)) * 2.0 * expected_value;
}

static long sample_gaussian(unsigned expected_value, unsigned variance)
{
    double u = sample_uniform(1) - 1;
    double v = sample_uniform(1) - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) {
        return sample_gaussian(expected_value, variance);
    }

    return (variance * u * sqrt(-2 * log(r) / r)) + expected_value;
}

#ifdef _WITH_UCX_
#define HAVE___CLEAR_CACHE 1
#include <ucs/time/time.h>
#define CURRENT_TIMER (ucs_time_interval_to_nsec(ucs_get_time()))
#else
#define CURRENT_TIMER (0)
#endif

void apply_imbalance(enum distribution_type imbalance,
                     unsigned imbalance_expected,
                     unsigned imbalance_variance)
{
    long wait_time_ns;
    struct timespec sleep_req;

    switch (imbalance) {
        case UNIFORM:
            wait_time_ns = sample_uniform(imbalance_expected);
            break;

        case GAUSSIAN:
            wait_time_ns = sample_gaussian(imbalance_expected, imbalance_variance);
            break;

        default:
            return;
    }

    double target = CURRENT_TIMER + wait_time_ns;
    while (CURRENT_TIMER < target);
}

/* vi:set sw=4 sts=4 tw=80: */
