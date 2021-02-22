./test_smallE_largeP.sh &> smallE_largeP.txt
./test_largeE_smallP.sh &> largeE_smallP.txt
./test_largeE_largeP.sh &> largeE_largeP.txt
python output_convert.py smallE_largeP.txt smallE_largeP_rebuild.dat smallE_largeP_push.dat
python output_convert.py largeE_smallP.txt largeE_smallP_rebuild.dat largeE_smallP_push.dat
python output_convert.py largeE_largeP.txt largeE_largeP_rebuild.dat largeE_largeP_push.dat
