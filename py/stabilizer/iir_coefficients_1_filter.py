#!/usr/bin/python3
"""
Authors:
    Étienne Wodey, Leibniz University Hannover, Institute of Quantum Optics
    Ryan Summers, Vertigo Designs
    Robert Jördens, QUARTIQ

Description: Algorithms to generate biquad (second order IIR) coefficients.
"""
import argparse
import asyncio
import collections
import logging

from math import pi, inf

import miniconf

import stabilizer

logger = logging.getLogger(__name__)

# Disable pylint warnings about a0, b1 etc
# pylint: disable=invalid-name


# Generic type for containing a command-line argument.
# Use `add_argument` for simple construction.
Argument = collections.namedtuple("Argument", ["positionals", "keywords"])


def add_argument(*args, **kwargs):
    """Convert arguments into an Argument tuple."""
    return Argument(args, kwargs)


# Represents a generic filter that can be implemented by a biquad.
#
# Fields
#     * `help`: This field specifies helpful human-readable information that
#       will be presented to users on the command line.
#     * `arguments`: A list of `Argument` objects representing available
#       command-line arguments for the filter. Use the `add_argument()`
#       function to easily parse options as they would be provided to
#       argparse.
#     * `coefficients`: A function, provided with parsed arguments, that
#       returns the IIR coefficients. See below for more information on this
#       function.
#
# # Coefficients Calculation Function
#     Description:
#       This function takes in two input arguments and returns the IIR filter
#       coefficients for Stabilizer to represent the necessary filter.
#
#     Args:
#       args: The filter command-line arguments. Any filter-related arguments
#       may be accessed via their name. E.g. `args.K`.
#
#     Returns:
#       [b0, b1, b2, -a1, -a2] IIR coefficients to be programmed into a
#       Stabilizer IIR filter configuration.
Filter = collections.namedtuple("Filter", ["help", "arguments", "coefficients"])


def get_filters():
    """Get a dictionary of all available filters.

    Note:
        Calculations coefficient largely taken using the derivations in
        page 9 of https://arxiv.org/pdf/1508.06319.pdf

        PII/PID coefficient equations are taken from the PID-IIR primer
        written by Robert Jördens at https://hackmd.io/IACbwcOTSt6Adj3_F9bKuw
    """
    return {
        "lowpass": Filter(
            help="Gain-limited low-pass filter",
            arguments=[
                add_argument(
                    "--f0", required=True, type=float, help="Corner frequency (Hz)"
                ),
                add_argument(
                    "--K", required=True, type=float, help="Lowpass filter gain"
                ),
            ],
            coefficients=lowpass_coefficients,
        ),
        "highpass": Filter(
            help="Gain-limited high-pass filter",
            arguments=[
                add_argument(
                    "--f0", required=True, type=float, help="Corner frequency (Hz)"
                ),
                add_argument(
                    "--K", required=True, type=float, help="Highpass filter gain"
                ),
            ],
            coefficients=highpass_coefficients,
        ),
        "allpass": Filter(
            help="Gain-limited all-pass filter",
            arguments=[
                add_argument(
                    "--f0", required=True, type=float, help="Corner frequency (Hz)"
                ),
                add_argument(
                    "--K", required=True, type=float, help="Allpass filter gain"
                ),
            ],
            coefficients=allpass_coefficients,
        ),
        "notch": Filter(
            help="Notch filter",
            arguments=[
                add_argument(
                    "--f0", required=True, type=float, help="Corner frequency (Hz)"
                ),
                add_argument(
                    "--Q", required=True, type=float, help="Filter quality factor"
                ),
                add_argument("--K", required=True, type=float, help="Filter gain"),
            ],
            coefficients=notch_coefficients,
        ),
        "pid": Filter(
            help="PID controller. Gains at 1 Hz and often negative.",
            arguments=[
                add_argument(
                    "--Kii", default=0, type=float, help="Double Integrator (I^2) gain"
                ),
                add_argument(
                    "--Kii_limit", default=inf, type=float, help="Integral gain limit"
                ),
                add_argument("--Ki", default=0, type=float, help="Integrator (I) gain"),
                add_argument(
                    "--Ki_limit", default=inf, type=float, help="Integral gain limit"
                ),
                add_argument(
                    "--Kp", default=0, type=float, help="Proportional (P) gain"
                ),
                add_argument("--Kd", default=0, type=float, help="Derivative (D) gain"),
                add_argument(
                    "--Kd_limit", default=inf, type=float, help="Derivative gain limit"
                ),
                add_argument(
                    "--Kdd", default=0, type=float, help="Double Derivative (D^2) gain"
                ),
                add_argument(
                    "--Kdd_limit", default=inf, type=float, help="Derivative gain limit"
                ),
            ],
            coefficients=pid_coefficients,
        ),
    }


def lowpass_coefficients(args):
    """Return low-pass IIR filter configuration."""
    return {
        "typ": "Filter",
        "repr": {
            "Filter": {
                "typ": "Lowpass",
                "frequency": args.f0,
                "gain": args.K,
            }
        }
    }, args.K


def highpass_coefficients(args):
    """Return high-pass IIR filter configuration."""
    return {
        "typ": "Filter",
        "repr": {
            "Filter": {
                "typ": "Highpass",
                "frequency": args.f0,
                "gain": args.K,
            }
        }
    }, args.K


def allpass_coefficients(args):
    """Return all-pass IIR filter configuration."""
    return {
        "typ": "Filter",
        "repr": {
            "Filter": {
                "typ": "Allpass",
                "frequency": args.f0,
                "gain": args.K,
            }
        }
    }, args.K


def notch_coefficients(args):
    """Return notch IIR filter configuration."""
    return {
        "typ": "Filter",
        "repr": {
            "Filter": {
                "typ": "Notch",
                "frequency": args.f0,
                "gain": args.K,
                "shape": {"Q": args.Q},
            }
        }
    }, args.K


def pid_coefficients(args):
    """Return PID IIR filter configuration."""
    if args.Kii != 0:
        order = "I2"
    elif args.Ki != 0:
        order = "I"
    else:
        order = "P"

    return {
        "typ": "Pid",
        "repr": {
            "Pid": {
                "order": order,
                "gain": {
                    "i2": args.Kii,
                    "i": args.Ki,
                    "p": args.Kp,
                    "d": args.Kd,
                    "d2": args.Kdd,
                },
                "limit": {
                    "i2": args.Kii_limit if args.Kii_limit != float("inf") else None,
                    "i": args.Ki_limit if args.Ki_limit != float("inf") else None,
                    "d": args.Kd_limit if args.Kd_limit != float("inf") else None,
                    "d2": args.Kdd_limit if args.Kdd_limit != float("inf") else None,
                }
            }
        }
    }, args.Kp


def _main():
    parser = argparse.ArgumentParser(
        description="Configure Stabilizer dual-iir filter parameters."
        "Note: This script assumes an AFE input gain of 1."
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase logging verbosity"
    )
    parser.add_argument(
        "--broker",
        "-b",
        type=str,
        default="192.168.199.251",
        help="The MQTT broker to use to communicate with " "Stabilizer (%(default)s)",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        default="dt/sinara/dual-iir/+",
        help="The Stabilizer device prefix in MQTT, "
        "wildcards allowed as long as the match is unique "
        "(%(default)s)",
    )
    parser.add_argument(
        "--no-discover",
        "-d",
        action="store_true",
        help="Do not discover Stabilizer device prefix.",
    )

    parser.add_argument(
        "--channel",
        "-c",
        type=int,
        choices=[0, 1],
        required=True,
        help="The filter channel to configure.",
    )
    parser.add_argument(
        "--sample-period",
        type=float,
        default=stabilizer.SAMPLE_PERIOD,
        help="Sample period in seconds (%(default)s s)",
    )

    parser.add_argument(
        "--y-min",
        type=float,
        default=-stabilizer.DAC_FULL_SCALE,
        help="The channel minimum output (%(default)s V)",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=stabilizer.DAC_FULL_SCALE,
        help="The channel maximum output (%(default)s V)",
    )
    parser.add_argument(
        "--cpu-dac1", type=int, default=2048, help="CPU DAC1 value (%(default)s)"
    )
    parser.add_argument(
        "--frontend-offset", type=int, default=0, help="Frontend offset (%(default)s)"
    )

    # Next, add subparsers and their arguments.
    subparsers = parser.add_subparsers(
        help="Filter-specific design parameters", dest="filter_type", required=True
    )

    filters = get_filters()

    for filter_name, filt in filters.items():
        subparser = subparsers.add_parser(filter_name, help=filt.help)
        for arg in filt.arguments:
            subparser.add_argument(*arg.positionals, **arg.keywords)

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.WARN - 10 * args.verbose,
    )

    # Calculate the IIR coefficients for the filter.
    config, forward_gain = filters[args.filter_type].coefficients(args)

    # The feed-forward gain of the IIR filter
    typ = config["typ"]
    inner = config["repr"][typ]
    if typ == "Filter":
        inner["offset"] = 0.0
    elif typ == "Pid":
        inner["setpoint"] = 0.0
    inner["min"] = stabilizer.voltage_to_machine_units(args.y_min)
    inner["max"] = stabilizer.voltage_to_machine_units(args.y_max)

    async def configure():
        async with miniconf.Client(
            args.broker,
            protocol=miniconf.MQTTv5,
            logger=logging.getLogger("aiomqtt-client"),
        ) as client:
            if not args.no_discover:
                prefix, _alive = await miniconf.discover_one(client, args.prefix)
            else:
                prefix = args.prefix

            interface = miniconf.Miniconf(client, prefix)

            # Set the filter coefficients.
            # Note: In the future, we will need to Handle higher-order cascades.
            await interface.set(
                f"/dual_iir/ch/{args.channel}/biquad/0",
                config,
            )
            print(f"Set filter representation: {config}")
            await interface.set(
                path="/dual_iir/cpu_dac1",
                value=args.cpu_dac1,
            )
            await interface.set(
                path="/dual_iir/frontend_offset",
                value=args.frontend_offset,
            )

    asyncio.run(configure())


if __name__ == "__main__":
    import os
    import sys

    if sys.platform.lower() == "win32" or os.name.lower() == "nt":
        from asyncio import set_event_loop_policy, WindowsSelectorEventLoopPolicy

        set_event_loop_policy(WindowsSelectorEventLoopPolicy())

    _main()
