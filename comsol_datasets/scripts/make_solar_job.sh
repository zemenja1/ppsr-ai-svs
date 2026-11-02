#!/usr/bin/env bash
# Writes a COMSOL Java batch class that computes J-V curves of the 1D silicon solar cell
# (Application Library: si_solar_cell_1d; n-type base, p+ emitter at the illuminated front x=0).
# Usage: make_solar_job.sh <ClassName> <devices.txt> <base.mph> <out.java>
# devices.txt lines: id;Nbase_cm3;Nemit_cm3;xj_um;W_um;tau_s;suns;T_K
# Output lines on stdout: JV,id,inputs...,V_V,I_A  plus START/DONE/FAIL markers.
cls="$1"; dev="$2"; base="$3"; out="$4"
{
cat <<EOF
import com.comsol.model.*;
import com.comsol.model.util.*;
public class $cls {
  static final String BASE = "$base";
  static final String[][] DEV = {
EOF
while IFS=';' read -r id nb ne xj w tau suns t; do
  [ -z "$id" ] && continue
  echo "    {\"$id\",\"$nb\",\"$ne\",\"$xj\",\"$w\",\"$tau\",\"$suns\",\"$t\"},"
done < "$dev"
cat <<'EOF'
  };
  static final String VLIST = "0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.47 0.49 0.51 0.53 0.55 0.57 0.59 0.61 0.63 0.65 0.67 0.69 0.71 0.73 0.75 0.77";
  static void out(String s) { System.out.println(s); System.out.flush(); }
  public static Model run() {
    Model m;
    try { m = ModelUtil.load("m", BASE); } catch (java.io.IOException e) { throw new RuntimeException(e); }
    // Require a small residual as well: with the default criterion COMSOL can accept a stalled Newton
    // iteration (tiny solution update, large residual) and return a wrong current without any warning.
    for (String sol : m.sol().tags())
      for (String f : m.sol(sol).feature().tags())
        for (String sub : m.sol(sol).feature(f).feature().tags())
          if (m.sol(sol).feature(f).feature(sub).getType().equals("FullyCoupled"))
            m.sol(sol).feature(f).feature(sub).set("termonres", "on");
    String gph = m.component("comp1").variable("var1").get("G_ph");
    StudyFeature s = m.study("std1").feature("stat");
    s.set("useparam", true);
    s.set("pname", new String[]{"V0"}); s.set("punit", new String[]{"V"});
    s.set("plistarr", new String[]{VLIST});
    s.set("pcontinuation", "V0");
    m.component("comp1").physics("semi").feature("tar1").set("taun_mat", "userdef");   // default: lifetime from the Si material
    m.component("comp1").physics("semi").feature("tar1").set("taup_mat", "userdef");
    NumericalFeature g = m.result().numerical().create("evJV", "EvalGlobal");
    g.set("data", "dset1");
    g.set("expr", new String[]{"V0", "semi.I0_1"});
    g.set("unit", new String[]{"V", "A"});
    for (String[] d : DEV) {
      long t0 = System.currentTimeMillis();
      out("START," + d[0]);
      try {
        m.component("comp1").physics("semi").feature("adm1").set("NDc", d[1] + "[1/cm^3]");
        m.component("comp1").physics("semi").feature("gdm1").set("NAgen", d[2] + "[1/cm^3]");
        m.component("comp1").physics("semi").feature("gdm1").set("jd_gen", d[3] + "[um]");
        m.component("comp1").geom("geom1").feature("i1").set("coord", new String[]{"0", d[4]});
        m.component("comp1").geom("geom1").run();
        m.component("comp1").mesh("mesh1").run();
        m.component("comp1").physics("semi").feature("tar1").set("taun", d[5] + "[s]");
        m.component("comp1").physics("semi").feature("tar1").set("taup", d[5] + "[s]");
        m.component("comp1").variable("var1").set("G_ph", "(" + d[6] + ")*(" + gph + ")");
        m.component("comp1").physics("semi").feature("smm1").set("minput_temperature", d[7] + "[K]");
        m.study("std1").run();
        double[][] r = g.getReal();
        for (int i = 0; i < r[0].length; i++)
          out(String.format(java.util.Locale.ROOT, "JV,%s,%s,%s,%s,%s,%s,%s,%s,%.6g,%.9e", d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], r[0][i], r[1][i]));
        out(String.format(java.util.Locale.ROOT, "DONE,%s,%d,%.1f", d[0], r[0].length, (System.currentTimeMillis() - t0) / 1000.0));
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
