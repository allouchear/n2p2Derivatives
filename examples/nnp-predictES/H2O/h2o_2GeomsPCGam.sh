mkdir /home/allouche/MySoftwares/CChemI/CChemI-05042019/cchemi/tests/HDNNP/h2o_2Geomstmp1
cd /home/allouche/MySoftwares/CChemI/CChemI-05042019/cchemi/tests/HDNNP/h2o_2Geomstmp1
cp /home/allouche/MySoftwares/CChemI/CChemI-05042019/cchemi/tests/HDNNP/h2o_2GeomsFF_1.inp input
firefly -p -o /home/allouche/MySoftwares/CChemI/CChemI-05042019/cchemi/tests/HDNNP/h2o_2GeomsFF_1.log
cd ..
mv PUNCH  /home/allouche/MySoftwares/CChemI/CChemI-05042019/cchemi/tests/HDNNP/h2o_2GeomsFF_1.pun
/bin/rm -r  /home/allouche/MySoftwares/CChemI/CChemI-05042019/cchemi/tests/HDNNP/h2o_2Geomstmp1
mkdir /home/allouche/MySoftwares/CChemI/CChemI-05042019/cchemi/tests/HDNNP/h2o_2Geomstmp2
cd /home/allouche/MySoftwares/CChemI/CChemI-05042019/cchemi/tests/HDNNP/h2o_2Geomstmp2
cp /home/allouche/MySoftwares/CChemI/CChemI-05042019/cchemi/tests/HDNNP/h2o_2GeomsFF_2.inp input
firefly -p -o /home/allouche/MySoftwares/CChemI/CChemI-05042019/cchemi/tests/HDNNP/h2o_2GeomsFF_2.log
cd ..
mv PUNCH  /home/allouche/MySoftwares/CChemI/CChemI-05042019/cchemi/tests/HDNNP/h2o_2GeomsFF_2.pun
/bin/rm -r  /home/allouche/MySoftwares/CChemI/CChemI-05042019/cchemi/tests/HDNNP/h2o_2Geomstmp2
