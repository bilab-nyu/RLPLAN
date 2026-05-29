import csv
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from PIL import Image


@dataclass
class SpatialIntegrationConfig:
    enabled: bool
    work_dir: str
    depthmapx_cli: str
    qgis_process: str
    qgis_env_bat: str = ""
    reward_weight: float = 1.0
    integration_column: str = "Visual Integration [HH]"
    timeout_seconds: int = 120
    keep_artifacts: bool = False
    fail_on_error: bool = False


class DepthmapIntegrationEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate(self, screen_array):
        if not self.config.enabled:
            return self._result("disabled")

        missing = self._missing_runtime_dependencies()
        if missing:
            return self._result("skipped", reason=", ".join(missing))

        run_dir = self._create_run_dir()
        try:
            dxf_path = self._export_layout_to_dxf(screen_array, run_dir)
            csv_path = self._run_depthmap_vga(dxf_path, run_dir)
            metrics = self._read_integration_metrics(csv_path)
            reward = metrics["mean"] * self.config.reward_weight
            metrics.update(
                {
                    "status": "ok",
                    "reward": reward,
                    "work_dir": run_dir if self.config.keep_artifacts else None,
                }
            )
            return metrics
        except Exception as exc:
            if self.config.fail_on_error:
                raise
            return self._result("failed", reason=str(exc))
        finally:
            if not self.config.keep_artifacts:
                shutil.rmtree(run_dir, ignore_errors=True)

    def _result(self, status, reason=None):
        result = {
            "status": status,
            "reward": 0.0,
            "mean": None,
            "max": None,
            "count": 0,
        }
        if reason:
            result["reason"] = reason
        return result

    def _missing_runtime_dependencies(self):
        missing = []

        try:
            from osgeo import gdal, ogr, osr  # noqa: F401
        except ImportError:
            missing.append("GDAL Python bindings are not installed")

        if not self._command_exists(self.config.qgis_process):
            missing.append(f"QGIS process not found: {self.config.qgis_process}")

        if not self._command_exists(self.config.depthmapx_cli):
            missing.append(f"depthmapX CLI not found: {self.config.depthmapx_cli}")

        return missing

    def _command_exists(self, command):
        if not command:
            return False
        if os.path.isabs(command) or os.path.dirname(command):
            return os.path.exists(command)
        return shutil.which(command) is not None

    def _create_run_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        run_dir = os.path.join(self.config.work_dir, f"integration_{timestamp}")
        os.makedirs(run_dir, exist_ok=False)
        return run_dir

    def _export_layout_to_dxf(self, screen_array, output_dir):
        from osgeo import gdal, ogr, osr

        screen_array = np.asarray(screen_array)
        raw_png = os.path.join(output_dir, "layout.png")
        Image.fromarray(screen_array).save(raw_png)

        gray = cv2.cvtColor(screen_array, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        external_wall = np.zeros_like(binary)
        cv2.drawContours(external_wall, contours, -1, 255, 1)

        mask_path = os.path.join(output_dir, "layout_mask.png")
        Image.fromarray(external_wall).save(mask_path)

        shapefile_path = os.path.join(output_dir, "layout.shp")
        raster_ds = gdal.Open(mask_path, gdal.GA_ReadOnly)
        source_band = raster_ds.GetRasterBand(1)

        driver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(shapefile_path):
            driver.DeleteDataSource(shapefile_path)

        shapefile_ds = driver.CreateDataSource(shapefile_path)
        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromEPSG(4326)
        layer = shapefile_ds.CreateLayer("layout", srs=spatial_ref)
        layer.CreateField(ogr.FieldDefn("ID", ogr.OFTInteger))
        gdal.Polygonize(source_band, None, layer, 0, [], callback=None)

        shapefile_ds = None
        raster_ds = None

        dxf_path = os.path.join(output_dir, "layout.dxf")
        self._run_qgis_dxf_export(shapefile_path, dxf_path, output_dir)
        return dxf_path

    def _run_qgis_dxf_export(self, shapefile_path, dxf_path, work_dir):
        args = [
            self.config.qgis_process,
            "run",
            "native:dxfexport",
            "--",
            f"LAYERS={shapefile_path}",
            "SYMBOLOGY_MODE=0",
            "SYMBOLOGY_SCALE=1",
            "ENCODING=cp1252",
            'CRS=QgsCoordinateReferenceSystem("EPSG:4326")',
            f"OUTPUT={dxf_path}",
        ]

        if self.config.qgis_env_bat and os.path.exists(self.config.qgis_env_bat):
            command = f'call "{self.config.qgis_env_bat}" && {subprocess.list2cmdline(args)}'
            subprocess.run(
                command,
                cwd=work_dir,
                check=True,
                shell=True,
                timeout=self.config.timeout_seconds,
            )
        else:
            subprocess.run(
                args,
                cwd=work_dir,
                check=True,
                timeout=self.config.timeout_seconds,
            )

    def _run_depthmap_vga(self, dxf_path, work_dir):
        graph_path = os.path.join(work_dir, "layout.graph")
        pre_graph_path = os.path.join(work_dir, "layout_pre.graph")
        visibility_graph_path = os.path.join(work_dir, "layout_visibility.graph")
        csv_path = os.path.join(work_dir, "integration.csv")

        commands = [
            [self.config.depthmapx_cli, "-m", "IMPORT", "-f", dxf_path, "-o", graph_path],
            [
                self.config.depthmapx_cli,
                "-m",
                "VISPREP",
                "-f",
                graph_path,
                "-o",
                pre_graph_path,
                "-pg",
                "1",
                "-pp",
                "25,25",
                "-pm",
            ],
            [
                self.config.depthmapx_cli,
                "-m",
                "VGA",
                "-vm",
                "visibility",
                "-vg",
                "n",
                "-vl",
                "-vr",
                "n",
                "-f",
                pre_graph_path,
                "-o",
                visibility_graph_path,
            ],
            [
                self.config.depthmapx_cli,
                "-m",
                "EXPORT",
                "-em",
                "pointmap-data-csv",
                "-o",
                csv_path,
                "-f",
                visibility_graph_path,
            ],
        ]

        for command in commands:
            subprocess.run(
                command,
                cwd=work_dir,
                check=True,
                timeout=self.config.timeout_seconds,
            )

        return csv_path

    def _read_integration_metrics(self, csv_path):
        values = []
        with open(csv_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if self.config.integration_column not in reader.fieldnames:
                raise ValueError(f"Missing CSV column: {self.config.integration_column}")

            for row in reader:
                value = row.get(self.config.integration_column, "")
                if value:
                    values.append(float(value))

        if not values:
            raise ValueError("No integration values were exported by depthmapX")

        return {
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }
