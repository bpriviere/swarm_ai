#!/bin/bash
# Test script for automated testing of everything

# Individual file processing
echo Testing individual file processing
python3 plotter.py ../current_results/sim_result_0.pickle --outputPDF ../plots/individual_test_pdf.pdf
python3 plotter.py ../current_results/sim_result_0.pickle --outputMP4 ../plots/individual_test_mp4.mp4

# Batch file processing
echo Testing batch file processing
python3 plotter.py ../current_results/ --outputPDF ../plots/group_test_pdf.pdf --outputMP4 ../plots/group_test_mp4.mp4
