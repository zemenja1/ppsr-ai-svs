# Latin hypercube sample printer.
# Usage: awk -v n=150 -v seed=42 -v spec="NA:log:1e16:3e17,tox:lin:10:40" -f lhs.awk
# Prints n rows of space-separated values, one column per spec entry, in random row order.
BEGIN {
  srand(seed)
  nd = split(spec, dims, ",")
  for (d = 1; d <= nd; d++) {
    split(dims[d], f, ":")
    kind[d] = f[2]; lo[d] = f[3] + 0; hi[d] = f[4] + 0
    for (i = 1; i <= n; i++) perm[i] = i
    for (i = n; i > 1; i--) { j = int(rand() * i) + 1; t = perm[i]; perm[i] = perm[j]; perm[j] = t }
    for (i = 1; i <= n; i++) {
      u = (perm[i] - 1 + rand()) / n
      if (kind[d] == "log") v = exp(log(lo[d]) + u * (log(hi[d]) - log(lo[d])))
      else v = lo[d] + u * (hi[d] - lo[d])
      val[i, d] = v
    }
  }
  for (i = 1; i <= n; i++) {
    line = ""
    for (d = 1; d <= nd; d++) line = line (d > 1 ? " " : "") sprintf("%.4g", val[i, d])
    print line
  }
}
