#!/bin/bash

# remove lines if weight == 4
awk '{if (substr($0,14,3)!=" 4 ") print $0}' ahar-var_new.out > ahar-var_corr.out

# locate using hypocenter
hyp << EOF
ahar-var_corr.out
n
EOF

# select events with rms < 0.5, erh < 10.0 erz < 20.0
/home/saeed/Programs/seisan/PRO/select select.inp

# locate using hypocenter
hyp << EOF
select.out
n
EOF

# remove unnecessary files
rm final.out hypmag.out hypsum.out index.codaq index.out waveform_names.out gmap.cur.kml

