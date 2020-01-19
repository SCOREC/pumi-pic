debug=0
C=32
# laptop can support up to ~90M particles before memory compression is used
#limit=$((90*1000*1000))
# blockade can support up to ~150M particles before memory is exhausted on the GPU
limit=$((150*1000*1000))
for dist in {1..3}; do
  for V in 8 16 32 64; do
    e=17;
    #for e in 11 14 17; do
      p=10
      log="d${dist}_e${e}_p${p}_C${C}_V${V}_sorted.log"
      cat /dev/null > $log
      elms=$((2**e))
      sigma=$elms #full sorting
      particles=$((2**p*elms))
      echo "dist $dist elements $elms particles/elm $((2**p)) total_particles $particles"
      if [ $particles -lt $limit ]; then
        ./pskk $elms $particles $dist $C $sigma $V $debug &>> $log
      else
        echo "skipping.. limit exceeded!"
      fi
    #done
  done
done
