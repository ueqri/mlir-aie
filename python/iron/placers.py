# placers.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from abc import ABCMeta, abstractmethod
import statistics
import json
import numpy as np
import os
import sys
from subprocess import Popen, PIPE, TimeoutExpired
from dataclasses import dataclass

from .device import Device
from .runtime import Runtime, RuntimeEndpoint
from .worker import Worker
from .device import AnyComputeTile, AnyMemTile, AnyShimTile, Tile
from .dataflow import ObjectFifoHandle, ObjectFifoLink, ObjectFifoEndpoint

@dataclass
class CommandResult:
    cmd: str
    cwd: str
    env: dict
    stdout: str
    stderr: str
    returncode: int
    is_timed_out: bool

    def ok(self):
        return self.returncode == 0 and not self.is_timed_out

    def check(self):
        if not self.ok():
            raise RuntimeError(
                f"Command '{self.cmd}' (on directory '{self.cwd}' with env '{self.env}') "
                f"failed with return code {self.returncode}.\n\n"
                f"Stdout:\n{self.stdout}\n\nStderr:\n{self.stderr}"
            )

    def __repr__(self):
        return (
            f"CommandResult(cmd={self.cmd}, cwd={self.cwd}, returncode={self.returncode}, "
            f"is_timed_out={self.is_timed_out},\nstdout={self.stdout},\nstderr={self.stderr},\nenv={self.env})"
        )

def read_text_file(file_path):
    with open(file_path, "r") as f:
        return f.read()


def write_text_file(file_path, content):
    with open(file_path, "w") as f:
        f.write(content)

def subprocess_run_cmd(cmd, cwd=None, env=None, timeout_sec=36000):
    assert isinstance(cmd, str), "Command must be a string"
    process = Popen(cmd, cwd=cwd, shell=True, stdout=PIPE, stderr=PIPE, env=env)

    is_timed_out = False

    try:
        stdout, stderr = process.communicate(timeout=timeout_sec)
    except TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        is_timed_out = True
        print(f"Command '{cmd}' timed out after {timeout_sec} seconds.")

    return CommandResult(
        cmd=cmd,
        cwd=cwd,
        env=env,
        stdout=stdout.decode(),
        stderr=stderr.decode(),
        returncode=process.returncode,
        is_timed_out=is_timed_out,
    )


class Placer(metaclass=ABCMeta):
    """Placer is an abstract class to define the interface between the Program
    and the Placer.
    """

    @abstractmethod
    def make_placement(
        self,
        device: Device,
        rt: Runtime,
        workers: list[Worker],
        object_fifos: list[ObjectFifoHandle],
    ):
        """Assign placement informatio to a program.

        Args:
            device (Device): The device to use for placement.
            rt (Runtime): The runtime information for the program.
            workers (list[Worker]): The workers included in the program.
            object_fifos (list[ObjectFifoHandle]): The object fifos used by the program.
        """
        ...

class SAPlacer(Placer):
    """
    Simulated Annealing Placer
    """

    def __init__(self, pnr_args: str = None):
        super().__init__()
        self._pnr_args = pnr_args
    def make_placement(
        self,
        device: Device,
        rt: Runtime,
        workers: list[Worker],
        fifo_handles: list[ObjectFifoHandle],
    ):

        data = {"nodes": [], "nets": [], "links": []}
        
        # Get FIFO list
        seen = set()
        of_list = []
        for ofh in fifo_handles:
            fifo = ofh._object_fifo
            if fifo not in seen:
                seen.add(fifo)
                of_list.append(fifo)

        fifo_ids = {of: idx for idx, of in enumerate(of_list)}
        id_to_fifo = {idx: of for idx, of in enumerate(of_list)}
        eps_to_ids: dict[ObjectFifoEndpoint, int] = {}
        ids_to_eps: dict[int, ObjectFifoEndpoint] = {}

        # Build nets
        for idx, of in enumerate(of_list):
            src_node_id = self._get_or_assign_id(eps_to_ids, ids_to_eps, of._prod.endpoint)
            dst_node_ids = [self._get_or_assign_id(eps_to_ids, ids_to_eps, c.endpoint) for c in of._cons]
            depths = of._get_depths()
            if not isinstance(depths, list):
                depths = [depths]
            depths = [int(d) for d in depths]  # ensure plain ints
            bytes_per_depth = int(np.prod(of.shape) * np.dtype(of.dtype).itemsize)

            data["nets"].append({
                "net_id": idx,
                "net_name": of.name,
                "src_id": src_node_id,
                "dst_ids": dst_node_ids,
                "depths": depths,
                "byte_size_per_depth": bytes_per_depth
            })

        # Build nodes
        for ep, n_id in eps_to_ids.items():
            t_type, row_y = self._get_node_info(ep.tile)
            data["nodes"].append({"id": n_id, "type": t_type, "col_x": -1, "row_y": row_y})

        # Build links
        link_set = set()
        for of in of_list:
            endpoints = of._get_endpoint(is_prod=True) + of._get_endpoint(is_prod=False)
            for ep in endpoints:
                if isinstance(ep, ObjectFifoLink):
                    link_set.add(ep)
        for link in link_set:
            src_handles = link._srcs if isinstance(link._srcs, list) else [link._srcs]
            dst_handles = link._dsts if isinstance(link._dsts, list) else [link._dsts]

            src_net_ids = sorted({fifo_ids[h._object_fifo] for h in src_handles})
            dst_net_ids = sorted({fifo_ids[h._object_fifo] for h in dst_handles})

            data["links"].append({
                "src_net_ids": src_net_ids,
                "dst_net_ids": dst_net_ids
            })

        with open("build/netlist.json", "w") as f:
            json.dump(data, f, indent=2)

        
        # --- Run placer subprocess here ---
        pnr_bin = os.path.expandvars("$NPU_PNR_BIN_DIR/placer")
        assert os.path.exists(pnr_bin), f"PnR binary not found at {pnr_bin}"
        pnr_cmd = str(
            f"{pnr_bin} build/netlist.json "
            "--output=build/pnr_placed_netlist.json "
            "--route-summary=build/pnr_route_summary.json "
            "--enable-packing "
        )
        if self._pnr_args is not None:
            pnr_cmd += f" {self._pnr_args}"
        print(f"Placing and routing: {pnr_cmd} ...", file=sys.stderr, flush=True)
        cwd = os.getcwd()
        pnr = subprocess_run_cmd(pnr_cmd, cwd=cwd)
        pnr.check()
        print("Finished placing and routing ...", file=sys.stderr, flush=True)
        # Load placed netlist
        with open("build/pnr_placed_netlist.json", "r") as f:
            placed_data = json.load(f)

        # Map coordinates to tiles
        coord_to_tile: dict[tuple[int, int], Tile] = {}
        for node in placed_data["nodes"]:
            coord = (node["col_x"], node["row_y"])
            t = coord_to_tile.get(coord)
            if t is None:
                t = Tile(*coord)
                coord_to_tile[coord] = t
            ids_to_eps[node["id"]].place(t)
        # assert every endpoint.tile is instance Tile
        for ep, n_id in eps_to_ids.items():
            if not isinstance(ep.tile, Tile):
                raise ValueError(f"Endpoint {ep} was not placed by placer!")
        # Apply net placements and routing info
        for net in placed_data["nets"]:
            of = id_to_fifo[net["net_id"]]

            routing = net["routing_info"]
            if routing["connection_type"] == "circuit_switch":
                of._via_dma = True
                of._hop_tile_ids = routing["intermediates"]
            elif routing["connection_type"] == "neighbor_sharing":
                alloc_tiles = []
                for tile_loc in routing["allocation_tiles"]:
                    coord = (tile_loc["col_x"], tile_loc["row_y"])
                    if coord not in coord_to_tile:
                        t = Tile(*coord)
                        coord_to_tile[coord] = t
                        device.resolve_tile(t)
                    alloc_tiles.append(coord_to_tile[coord])
                of._allocate(alloc_tiles)
            elif routing["connection_type"] == "intra_tile":
                # TODO: apply allocate to other connection types
                continue
            else:
                raise ValueError(f"Unknown connection type {routing['connection_type']}")
        # Place buffers for workers
        for worker in workers:
            assert isinstance(worker.tile, Tile), f"Worker {worker} was not placed by placer!"
            for buffer in worker.buffers:
                buffer.place(worker.tile)

    def _get_node_info(self, tile):
        if tile == AnyComputeTile:
            return "COMP", -1
        elif tile == AnyMemTile:
            return "MEM", 1
        elif tile == AnyShimTile:
            return "SHIM", 0
        elif isinstance(tile, Tile):
            if tile.row == 0:
                return "SHIM", 0
            elif tile.row == 1:
                return "MEM", 1
            else:
                return "COMP", -1
        else:
            raise ValueError("Unknown tile type." + str(tile))

    def _get_or_assign_id(
        self, 
        endpoint_to_ids, 
        ids_to_endpoints, 
        endpoint
    ):
        if endpoint not in endpoint_to_ids:
            new_id = len(endpoint_to_ids)
            endpoint_to_ids[endpoint] = new_id
            ids_to_endpoints[new_id] = endpoint
        return endpoint_to_ids[endpoint]

class SequentialPlacer(Placer):
    """SequentialPlacer is a simple implementation of a placer. The SequentialPlacer is so named
    because it will sequentially place workers to Compute Tiles. After workers are placed, Memory Tiles and
    Shim Tiles are placed as close to the column of the given compute tile as possible.

    The SequentialPlacer only does validation of placement with respect to available DMA channels on the tiles.
    However, it can yield invalid placements that exceed other resource limits, such as memory, For complex or
    resource sensitive designs, a more complex placer or manual placement is required.
    """

    def __init__(self):
        super().__init__()

    def make_placement(
        self,
        device: Device,
        rt: Runtime,
        workers: list[Worker],
        object_fifos: list[ObjectFifoHandle],
    ):
        # Keep track of tiles available for placement based
        # on number of available input / output DMA channels
        shims_in = device.get_shim_tiles()
        shims_out = device.get_shim_tiles()

        mems_in = device.get_mem_tiles()
        mems_out = device.get_mem_tiles()

        computes = device.get_compute_tiles()
        computes_in = device.get_compute_tiles()
        computes_out = device.get_compute_tiles()
        compute_idx = 0

        # For each tile keep track of how many input and output endpoints there are
        # Note: defaultdict(list) automatically assigns an empty list as the default value for
        # keys that don’t exist
        channels_in: dict[Tile, tuple[ObjectFifoEndpoint, int]] = {}
        channels_out: dict[Tile, tuple[ObjectFifoEndpoint, int]] = {}

        # If some workers are already taken, remove them from the available set
        for worker in workers:
            # This worker has already been placed
            if isinstance(worker.tile, Tile):
                if not worker.tile in computes:
                    raise ValueError(
                        f"Partial Placement Error: "
                        f"Tile {worker.tile} not available on "
                        f"device {device} or has already been used."
                    )
                computes.remove(worker.tile)

        for worker in workers:
            if worker.tile == AnyComputeTile:
                if compute_idx >= len(computes):
                    raise ValueError("Ran out of compute tiles for placement!")
                worker.place(computes[compute_idx])
                compute_idx += 1

            for buffer in worker.buffers:
                buffer.place(worker.tile)

            # Account for channels used by Workers, which are already placed
            prod_fifos = [of for of in worker.fifos if of._is_prod]
            cons_fifos = [of for of in worker.fifos if not of._is_prod]
            self._update_channels(
                worker,
                worker.tile,
                True,
                len(prod_fifos),
                channels_out,
                computes_out,
                device,
            )
            self._update_channels(
                worker,
                worker.tile,
                False,
                len(cons_fifos),
                channels_in,
                computes_in,
                device,
            )

        # Prepare to loop
        if len(computes) > 0:
            compute_idx = compute_idx % len(computes)

        for ofh in object_fifos:
            of_endpoints = ofh.all_of_endpoints()
            of_handle_endpoints = ofh._object_fifo._get_endpoint(is_prod=ofh._is_prod)
            of_compute_endpoints_tiles = [
                ofe.tile for ofe in of_endpoints if ofe.tile in computes
            ]
            common_col = self._get_common_col(of_compute_endpoints_tiles)
            of_link_endpoints = [
                ofe for ofe in of_endpoints if isinstance(ofe, ObjectFifoLink)
            ]
            # Place "closest" to the compute endpoints
            for ofe in of_handle_endpoints:
                if isinstance(ofe, Worker):
                    continue

                if ofe.tile == AnyMemTile:
                    if ofh._is_prod:
                        self._place_endpoint(
                            ofe,
                            mems_out,
                            common_col,
                            channels_out,
                            device,
                            output=True,
                        )
                    else:
                        self._place_endpoint(
                            ofe,
                            mems_in,
                            common_col,
                            channels_in,
                            device,
                        )

                elif ofe.tile == AnyShimTile:
                    if ofh._is_prod:
                        self._place_endpoint(
                            ofe,
                            shims_out,
                            common_col,
                            channels_out,
                            device,
                            output=True,
                        )
                    else:
                        self._place_endpoint(
                            ofe, shims_in, common_col, channels_in, device
                        )

            for ofe in of_link_endpoints:
                # When placing ObjectFifoLink endpoints account for both
                # input and output channel requirements
                if ofe.tile == AnyMemTile:
                    if ofh._is_prod:
                        self._place_endpoint(
                            ofe,
                            mems_out,
                            common_col,
                            channels_out,
                            device,
                            output=True,
                            link_tiles=mems_in,
                            link_channels=channels_in,
                        )
                    else:
                        self._place_endpoint(
                            ofe,
                            mems_in,
                            common_col,
                            channels_in,
                            device,
                            link_tiles=mems_out,
                            link_channels=channels_out,
                        )

                elif ofe.tile == AnyComputeTile:
                    if ofh._is_prod:
                        self._place_endpoint(
                            ofe,
                            computes_out,
                            common_col,
                            channels_out,
                            device,
                            output=True,
                            link_tiles=computes_in,
                            link_channels=channels_in,
                        )
                    else:
                        self._place_endpoint(
                            ofe,
                            computes_in,
                            common_col,
                            channels_in,
                            device,
                            link_tiles=computes_out,
                            link_channels=channels_out,
                        )

    def _get_common_col(self, tiles: list[Tile]) -> int:
        """
        A utility function that calculates a column that is "close" or "common"
        to a set of tiles. It is a simple heuristic using the average to represent "distance".
        """
        cols = [t.col for t in tiles if isinstance(t, Tile)]
        if len(cols) == 0:
            return 0
        avg_col = round(statistics.mean(cols))
        return avg_col

    def _find_col_match(self, col: int, tiles: list[Tile], device: Device) -> Tile:
        """
        A utility function that sequentially searches a list of tiles to find one with a matching column.
        The column is increased until a tile is found in the device, or an error is signaled.
        """
        new_col = col
        while new_col < device.cols:
            for t in tiles:
                if t.col == new_col:
                    return t
            new_col += 1
        raise ValueError(
            f"Failed to find a tile matching column {col}: tried until column {new_col}. Try using a device with more columns."
        )

    def _update_channels(
        self,
        ofe: ObjectFifoEndpoint,
        tile: Tile,
        output: bool,
        num_required_channels: int,
        channels: dict[Tile, tuple[ObjectFifoEndpoint, int]],
        tiles: list[Tile],
        device: Device,
    ):
        """
        A utility function that updates given channel and tile lists. It appends a new
        (endpoint, num_required_channels) entry to the channels dict for the given tile key, then
        verifies whether the total entries for that tile surpass the maximum number of available
        channels. If so, it removes the tile from the list of available tiles.
        """
        if num_required_channels == 0:
            return
        if tile not in channels:
            channels[tile] = []
        channels[tile].append((ofe, num_required_channels))
        used_channels = 0
        for _, c in channels[tile]:
            used_channels += c
        max_tile_channels = device.get_num_connections(tile, output)
        if used_channels >= max_tile_channels:
            tiles.remove(tile)

    def _place_endpoint(
        self,
        ofe: ObjectFifoEndpoint,
        tiles: list[Tile],
        common_col: int,
        channels: dict[Tile, tuple[ObjectFifoEndpoint, int]],
        device: Device,
        output=False,
        link_tiles=[],
        link_channels={},
    ):
        """
        A utility function that places a given endpoint based on available DMA channels. If the endpoint is a
        link, both input and output channels should be accounted for. Calls _update_channels() to update channel
        dictionaries and tile lists.
        """
        is_shim = False
        num_required_channels = 1
        if isinstance(ofe, ObjectFifoLink):
            # If endpoint is a link, account for both input and output DMA channels
            if output:
                num_required_channels = len(ofe._srcs)
                link_required_channels = len(ofe._dsts)
            else:
                num_required_channels = len(ofe._dsts)
                link_required_channels = len(ofe._srcs)

        # Check if placing is possible
        test_tiles = tiles.copy()
        while True:
            tile = self._find_col_match(common_col, test_tiles, device)
            total_channels = num_required_channels
            if tile in channels:
                for _, c in channels[tile]:
                    total_channels += c
            max_tile_channels = device.get_num_connections(tile, output)
            if total_channels <= max_tile_channels:
                if isinstance(ofe, ObjectFifoLink):
                    # Also check for channels in the other link direction
                    total_link_channels = link_required_channels
                    if tile in link_channels:
                        for _, c in link_channels[tile]:
                            total_link_channels += c
                    max_link_channels = device.get_num_connections(tile, not output)
                    if total_link_channels <= max_link_channels:
                        break
                else:
                    break
            test_tiles.remove(tile)

        # If no error was signaled by _find_col_match(), placement is possible
        ofe.place(tile)

        # Account for channels that were used by this placement
        self._update_channels(
            ofe,
            tile,
            output,
            num_required_channels,
            channels,
            tiles,
            device,
        )

        if isinstance(ofe, ObjectFifoLink):
            self._update_channels(
                ofe,
                tile,
                not output,
                link_required_channels,
                link_channels,
                link_tiles,
                device,
            )
