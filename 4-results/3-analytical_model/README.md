In Linux,


you can use the following one-liner:

cat trace-results.csv | grep ACC | cut -c7-12,14-16,18-21 | paste -d " "  - - | awk 'BEGIN { FS=OFS=" "; } { print $1, $2, ($2-$1)*1000}'

to get the accelerator-start, accelerator-end, and their difference times in ms

and

cat trace-results.csv | grep ACC | cut -c7-12,14-16,18-21 | paste -d " "  - - | awk 'BEGIN { FS=OFS=" "; } { print $1, $2, ($2-$1)*1000}' | cut -d" " -f3 >  > trace_latencies.txt

to get only the difference times ms.


you can use the following one-liner:

cat simulation-result.transaction.rpt | sed -n '1!p' | cut -c30-36 | awk 'BEGIN { FS=OFS=" "; } { print $1*0.00001}' > simulation_latencies.txt

to get the simulation latencies in ms.