#!/usr/bin/env bash
# Usage: make_worker.sh <ClassName> <devices.txt> <base.mph> <out.java>
# devices.txt lines: tier;dev;NA_cm3;tox_nm;Lg_um;xj_um;T_K;VG1 list;VG2 list;VD list
cls="$1"; dev="$2"; base="$3"; out="$4"
{
cat <<EOF
import com.comsol.model.*;
import com.comsol.model.util.*;
public class $cls {
  static final String BASE = "$base";
  static final String[][] DEV = {
EOF
while IFS=';' read -r tier id na tox lg xj t vg1 vg2 vd; do
  [ -z "$id" ] && continue
  echo "    {\"$tier\",\"$id\",\"$na\",\"$tox\",\"$lg\",\"$xj\",\"$t\",\"$vg1\",\"$vg2\",\"$vd\"},"
done < "$dev"
cat <<'EOF'
  };
  static void out(String s) { System.out.println(s); System.out.flush(); }
  static String hdr(String[] d) { return d[0]+","+d[1]+","+d[2]+","+d[3]+","+d[4]+","+d[5]+","+d[6]; }
  static int emit(Model m, String ds, String tag, String[] d) {
    NumericalFeature g = m.result().numerical("ev_" + ds);
    g.set("data", ds);
    double[][] r = g.getReal();
    for (int i = 0; i < r[0].length; i++)
      out(String.format(java.util.Locale.ROOT, "%s,%s,%.6g,%.6g,%.9e", tag, hdr(d), r[0][i], r[1][i], r[2][i]));
    return r[0].length;
  }
  public static Model run() {
    Model m;
    try { m = ModelUtil.load("m", BASE); } catch (java.io.IOException e) { throw new RuntimeException(e); }
    for (String ds : new String[]{"dset1", "dset2"}) {
      NumericalFeature g = m.result().numerical().create("ev_" + ds, "EvalGlobal");
      g.set("expr", new String[]{"Vg", "Vd", "semi.I0_2"});
      g.set("unit", new String[]{"V", "V", "A"});
    }
    for (String[] d : DEV) {
      long t0 = System.currentTimeMillis();
      out("START," + d[1]);
      try {
        double lg = Double.parseDouble(d[4]), xj = Double.parseDouble(d[5]);
        double g0 = 1.5 - lg / 2, g1 = 1.5 + lg / 2;
        m.component("comp1").geom("geom1").feature("pol1").set("table", new double[][]{
          {0, 0.67}, {0, 0.7}, {g0 - 0.2, 0.7}, {g0, 0.7}, {g1, 0.7}, {g1 + 0.2, 0.7}, {3, 0.7}, {3, 0.67}});
        m.component("comp1").geom("geom1").run();
        m.component("comp1").mesh("mesh1").run();
        m.component("comp1").physics("semi").feature("adm1").set("NAc", d[2] + "[1/cm^3]");
        String[] jda = {(0.8 * xj) + "[um]", xj + "[um]", "0"};
        String[] adm = {"adm2", "adm3"};
        String[] rbx = {"0[um]", (g1 + 0.1) + "[um]"};
        String[] wid = {(g0 - 0.1) + "[um]", (3 - g1 - 0.1) + "[um]"};
        for (int k = 0; k < 2; k++) {
          m.component("comp1").physics("semi").feature(adm[k]).set("rb", new String[]{rbx[k], "0.6[um]", "0"});
          m.component("comp1").physics("semi").feature(adm[k]).set("jwidth", wid[k]);
          m.component("comp1").physics("semi").feature(adm[k]).set("jda", jda);
        }
        m.component("comp1").physics("semi").feature("gc1").set("d_ins", d[3] + "[nm]");
        m.component("comp1").physics("semi").feature("smm1").set("minput_temperature", d[6] + "[K]");

        StudyFeature s1 = m.study("std1").feature("stat");
        s1.set("pname", new String[]{"Vd", "Vg"}); s1.set("punit", new String[]{"V", "V"});
        s1.set("plistarr", new String[]{"0.01", d[7]}); s1.set("pcontinuation", "Vg");
        m.study("std1").run();
        int n1 = emit(m, "dset1", "IDVG", d);

        String[] vg1 = d[7].trim().split(" +");
        String[] vg2 = d[8].trim().split(" +");
        StudyFeature s2 = m.study("std2").feature("stat");
        s2.set("pname", new String[]{"Vg", "Vd"}); s2.set("punit", new String[]{"V", "V"});
        s2.set("pcontinuation", "Vd");
        int n2 = 0;
        try {
          s2.set("plistarr", new String[]{d[8], d[9]});
          s2.set("solnum", String.valueOf(idx(vg1, vg2[0])));
          m.study("std2").run();
          n2 = emit(m, "dset2", "GRID", d);
        } catch (Exception e) {
          out("FALLBACK," + d[1] + "," + oneLine(e));
          for (String vg : vg2) {
            try {
              s2.set("plistarr", new String[]{vg, d[9]});
              s2.set("solnum", String.valueOf(idx(vg1, vg)));
              m.study("std2").run();
              n2 += emit(m, "dset2", "GRID", d);
            } catch (Exception e2) { out("FAILVG," + d[1] + "," + vg + "," + oneLine(e2)); }
          }
        }
        out(String.format(java.util.Locale.ROOT, "DONE,%s,%d,%d,%.1f", d[1], n1, n2, (System.currentTimeMillis() - t0) / 1000.0));
      } catch (Exception e) {
        out(String.format(java.util.Locale.ROOT, "FAIL,%s,%.1f,%s", d[1], (System.currentTimeMillis() - t0) / 1000.0, oneLine(e)));
      }
    }
    out("WORKER_FINISHED");
    return m;
  }
  static int idx(String[] list, String v) {
    double x = Double.parseDouble(v);
    for (int i = 0; i < list.length; i++) if (Math.abs(Double.parseDouble(list[i]) - x) < 1e-9) return i + 1;
    return 1;
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
