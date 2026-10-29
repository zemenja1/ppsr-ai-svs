#!/usr/bin/env bash
# Writes a COMSOL Java batch class that computes I-V curves of the 1D pn junction
# (Application Library: pn_junction_1d, finite volume study std2 / physics semi2).
# Usage: make_diode_job.sh <ClassName> <devices.txt> <base.mph> <out.java>
# devices.txt lines: id;Na_cm3;Nd_cm3;tau_s;T_K
# Output lines on stdout: IV,id,Na,Nd,tau,T,Va_V,I_A  plus START/DONE/FAIL markers.
cls="$1"; dev="$2"; base="$3"; out="$4"
{
cat <<EOF
import com.comsol.model.*;
import com.comsol.model.util.*;
public class $cls {
  static final String BASE = "$base";
  static final String[][] DEV = {
EOF
while IFS=';' read -r id na nd tau t; do
  [ -z "$id" ] && continue
  echo "    {\"$id\",\"$na\",\"$nd\",\"$tau\",\"$t\"},"
done < "$dev"
cat <<'EOF'
  };
  static final String FWD = "0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8";
  static final String REV = "0 -0.1 -0.25 -0.5 -1 -1.5 -2";
  static void out(String s) { System.out.println(s); System.out.flush(); }
  static int sweep(Model m, String list, String[] d, boolean skipZero) {
    m.study("std2").feature("stat").set("plistarr", new String[]{list});
    m.study("std2").run();
    double[][] r = m.result().numerical("evIV").getReal();
    int n = 0;
    for (int i = 0; i < r[0].length; i++) {
      if (skipZero && Math.abs(r[0][i]) < 1e-12) continue;
      out(String.format(java.util.Locale.ROOT, "IV,%s,%s,%s,%s,%s,%.6g,%.9e", d[0], d[1], d[2], d[3], d[4], r[0][i], r[1][i]));
      n++;
    }
    return n;
  }
  public static Model run() {
    Model m;
    try { m = ModelUtil.load("m", BASE); } catch (java.io.IOException e) { throw new RuntimeException(e); }
    StudyFeature s = m.study("std2").feature("stat");
    s.set("useparam", true);
    s.set("pname", new String[]{"Va"}); s.set("punit", new String[]{"V"});
    s.set("pcontinuation", "Va");
    NumericalFeature g = m.result().numerical().create("evIV", "EvalGlobal");
    g.set("data", "dset2");
    g.set("expr", new String[]{"Va", "semi2.I0_2"});
    g.set("unit", new String[]{"V", "A"});
    for (String[] d : DEV) {
      long t0 = System.currentTimeMillis();
      out("START," + d[0]);
      try {
        m.param().set("Na", d[1] + "[1/cm^3]");
        m.param().set("Nd", d[2] + "[1/cm^3]");
        m.param().set("Tl", d[4] + "[K]");
        m.component("comp1").physics("semi2").feature("tar1").set("taun", d[3] + "[s]");
        m.component("comp1").physics("semi2").feature("tar1").set("taup", d[3] + "[s]");
        int n = sweep(m, FWD, d, false);
        try { n += sweep(m, REV, d, true); } catch (Exception e) { out("FAILREV," + d[0] + "," + oneLine(e)); }
        out(String.format(java.util.Locale.ROOT, "DONE,%s,%d,%.1f", d[0], n, (System.currentTimeMillis() - t0) / 1000.0));
      } catch (Exception e) {
        out(String.format(java.util.Locale.ROOT, "FAIL,%s,%.1f,%s", d[0], (System.currentTimeMillis() - t0) / 1000.0, oneLine(e)));
      }
    }
    out("WORKER_FINISHED");
    return m;
  }
  static String oneLine(Exception e) {
    StringBuilder b = new StringBuilder();
    for (char c : String.valueOf(e.getMessage()).toCharArray()) b.append(c < 32 ? (char) 32 : (c == (char) 44 ? (char) 59 : c));
    String s = b.toString();
    return s.length() > 200 ? s.substring(0, 200) : s;
  }
  public static void main(String[] a) { run(); }
}
EOF
} > "$out"
