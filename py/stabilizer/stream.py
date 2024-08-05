#!/usr/bin/python3
"""Stabilizer streaming receiver and parsers"""

import argparse
import asyncio
import logging
import struct
import socket
import ipaddress
from collections import namedtuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np

import stabilizer

import stabilizer.iir_biquad_filter

# The number of DAC LSB codes per volt on Stabilizer outputs.
DAC_LSB_PER_VOLT = (1 << 16) / (4.096 * 5)

# The number of volts per ADC LSB.
ADC_VOLTS_PER_LSB = (5.0 / 2.0 * 4.096) / (1 << 15)

# The number of volts per DAC LSB.
DAC_VOLTS_PER_LSB = 1 / DAC_LSB_PER_VOLT

logger = logging.getLogger(__name__)

Trace = namedtuple("Trace", "values scale label")


def wrap(wide):
    """Wrap to 32 bit integer"""
    return wide & 0xFFFFFFFF


def get_local_ip(remote):
    """Get the local IP of a connection to the to a remote host.
    Returns a list of four octets."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((remote, 1883))
        return sock.getsockname()[0]
    finally:
        sock.close()


class AdcDac:
    """Stabilizer default striming data format"""

    format_id = 1

    def __init__(self, header, body):
        self.header = header
        self.body = body

    def size(self):
        """Return the data size of the frame in bytes"""
        return len(self.body)

    def to_mu(self):
        """Return the raw data in machine units"""
        data = np.frombuffer(self.body, "<i2")
        # batch, channel, sample
        data = data.reshape(self.header.batches, 4, -1)
        data = data.swapaxes(0, 1).reshape(4, -1)
        # convert DAC offset binary to two's complement
        data[2:] ^= np.uint16(0x8000)
        return data

    def to_si(self):
        """Convert the raw data to SI units"""
        data = self.to_mu() * DAC_VOLTS_PER_LSB
        return {
            "adc": data[:2],
            "dac": data[2:],
        }

    def to_traces(self):
        """Convert the raw data to labelled Trace instances"""
        data = self.to_mu()
        return [
            Trace(data[0], scale=DAC_VOLTS_PER_LSB, label="ADC0"),
            Trace(data[1], scale=DAC_VOLTS_PER_LSB, label="ADC1"),
            Trace(data[2], scale=DAC_VOLTS_PER_LSB, label="DAC0"),
            Trace(data[3], scale=DAC_VOLTS_PER_LSB, label="DAC1"),
        ]


class StabilizerStream(asyncio.DatagramProtocol):
    """Stabilizer streaming receiver protocol"""

    # The magic header half-word at the start of each packet.
    magic = 0x057B
    header_fmt = struct.Struct("<HBBI")
    header = namedtuple("Header", "magic format_id batches sequence")
    parsers = {
        AdcDac.format_id: AdcDac,
    }

    @classmethod
    async def open(cls, addr, port, broker, maxsize=1):
        """Open a UDP socket and start receiving frames"""
        loop = asyncio.get_running_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Increase the OS UDP receive buffer size to 4 MiB so that latency
        # spikes don't impact much. Achieving 4 MiB may require increasing
        # the max allowed buffer size, e.g. via
        # `sudo sysctl net.core.rmem_max=26214400` but nowadays the default
        # max appears to be ~ 50 MiB already.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)

        # We need to specify which interface to receive broadcasts from, or Windows may choose the
        # wrong one. Thus, use the broker address to figure out our local address for the interface
        # of interest.
        if ipaddress.ip_address(addr).is_multicast:
            print("Subscribing to multicast")
            group = socket.inet_aton(addr)
            iface = socket.inet_aton(".".join([str(x) for x in get_local_ip(broker)]))
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, group + iface)
            sock.bind(("", port))
        else:
            sock.bind((addr, port))

        transport, protocol = await loop.create_datagram_endpoint(
            lambda: cls(maxsize), sock=sock
        )
        return transport, protocol

    def __init__(self, maxsize):
        self.queue = asyncio.Queue(maxsize)

    def connection_made(self, _transport):
        logger.info("Connection made (listening)")

    def connection_lost(self, _exc):
        logger.info("Connection lost")

    def datagram_received(self, data, _addr):
        header = self.header._make(self.header_fmt.unpack_from(data))
        if header.magic != self.magic:
            logger.warning("Bad frame magic: %#04x, ignoring", header.magic)
            return
        try:
            parser = self.parsers[header.format_id]
        except KeyError:
            logger.warning("No parser for format %s, ignoring", header.format_id)
            return
        frame = parser(header, data[self.header_fmt.size :])
        if self.queue.full():
            old = self.queue.get_nowait()
            logger.debug("Dropping frame: %#08x", old.header.sequence)
        self.queue.put_nowait(frame)


async def measure(stream, duration):
    """Measure throughput and loss of stream reception, return the adc and dac data in SI units"""

    @dataclass
    class _Statistics:
        expect = None
        received = 0
        lost = 0
        bytes = 0

    stat = _Statistics()
    frames = []

    async def _record():
        nonlocal frames
        while True:
            frame = await stream.queue.get()
            if stat.expect is not None:
                stat.lost += wrap(frame.header.sequence - stat.expect)
            stat.received += frame.header.batches
            stat.expect = wrap(frame.header.sequence + frame.header.batches)
            stat.bytes += frame.size()
            frames.append(frame)

    try:
        await asyncio.wait_for(_record(), timeout=duration)
    except asyncio.TimeoutError:
        pass

    logger.info(
        "Received %g MB, %g MB/s", stat.bytes / 1e6, stat.bytes / 1e6 / duration
    )

    sent = stat.received + stat.lost
    if sent:
        loss = stat.lost / sent
    else:
        loss = 1
    logger.info("Loss: %s/%s batches (%g %%)", stat.lost, sent, loss * 1e2)

    # convert the frames into numpy arrays
    chucks_si = [chuck.to_si() for chuck in frames]
    adc1 = np.concatenate([chuck_si["adc"][0] for chuck_si in chucks_si])
    adc2 = np.concatenate([chuck_si["adc"][1] for chuck_si in chucks_si])
    dac1 = np.concatenate([chuck_si["dac"][0] for chuck_si in chucks_si])
    dac2 = np.concatenate([chuck_si["dac"][1] for chuck_si in chucks_si])

    return loss, adc1, adc2, dac1, dac2


async def main():
    """Test CLI"""
    parser = argparse.ArgumentParser(description="Stabilizer streaming demo")
    parser.add_argument(
        "--port", type=int, default=1234, help="Local port to listen on"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Local address to listen on")
    parser.add_argument(
        "--broker", default="192.168.199.251", help="The MQTT broker address"
    )
    parser.add_argument("--maxsize", type=int, default=1, help="Frame queue size")
    parser.add_argument("--duration", type=float, default=0.0001, help="Test duration")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    _transport, stream = await StabilizerStream.open(
        args.host, args.port, args.broker, args.maxsize
    )
    _loss, adc1, adc2, dac1, dac2 = await measure(stream, args.duration)

    print(adc1.shape, adc2.shape, dac1.shape, dac2.shape, "\n")

    # calculate the power spectral density
    fs = 1 / stabilizer.SAMPLE_PERIOD
    freq, psd = welch(adc1, fs, nperseg=256 * 16)
    t_s = np.arange(0, adc1.size) / fs

    iir_biquad_filter = stabilizer.iir_biquad_filter.IirBiquadFilter(
        "notch", f0=15.625e3, K=1, Q=10
    )

    # plot the data
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(t_s, adc1)
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Voltage [V]")
    axs[0, 0].set_title("ADC0")
    axs[0, 1].plot(np.round(freq / 1e3, 2), psd)
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_xlabel("Frequency [kHz]")
    axs[0, 1].set_ylabel("PSD [V**2/Hz]")
    axs[0, 1].set_title("PSD ADC0")
    axs[1, 1].plot(t_s, dac1)
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Voltage [V]")
    axs[1, 1].set_title("DAC0")
    axs[1, 0].plot(t_s, iir_biquad_filter.apply_filter(adc1))
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Voltage [V]")
    axs[1, 0].set_title("Filtered ADC0")
    plt.show()

    print(f"PSD sum up to 1kH: {np.sum(psd[freq < 10e3])}")


if __name__ == "__main__":
    asyncio.run(main())
