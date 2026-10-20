@echo off
echo Installing required Python packages...
python -m pip install trimesh scipy pillow opencv-python pyrender embreex

echo.
echo Running crosshair generation for ALL artifacts...
python photogrammetry\analyze_hsi_clusters.py --all

echo.
echo Done! Check photogrammetry\output\hsi_clusters\ for results.
pause