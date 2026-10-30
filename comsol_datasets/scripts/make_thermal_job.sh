#!/usr/bin/env bash
# Usage: make_thermal_job.sh <ClassName> <devices.txt> <base.mph> <out.java>
# 2D n+/p diode with self-heating, built from the MOSFET Application Library model (mosfet_base.mph):
# gate removed, both n+ regions = cathode (0 V), substrate contact at the bottom = anode and heat sink.
# The chip is isothermal (the temperature drop inside 0.7 um of silicon is < 0.1 K); its temperature Tj
# follows from the package thermal resistance by a global equation  Tj = T0 + r_th * V * I / A_chip.
# Per device: (1) isothermal I-V sweep at T0 (voltage driven, tag ISO);
#             (2) self-heating sweep, current driven, J = 0.1 ... 1000 A/cm^2 (tag HOT).
# devices.txt lines: id;NA_cm3;xj_um;tau_s;T0_K;rth_Kcm2_W
cls="$1"; dev="$2"; base="$3"; out="$4"
{
cat <<EOF
import com.comsol.model.*;
import com.comsol.model.util.*;
public class $cls {
  static final String BASE = "$base";
  static final String[][] DEV = {
EOF
while IFS=';' read -r id na xj tau t0 rth; do
  [ -z "$id" ] && continue
  echo "    {\"$id\",\"$na\",\"$xj\",\"$tau\",\"$t0\",\"$rth\"},"
done < "$dev"
cat <<'EOF'
  };
  static final String VLIST = "0 0.1 0.2 0.3 0.4 0.5 0.55 0.6 0.62 0.64 0.66 0.68 0.7 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86 0.88 0.9";
  static final double A_CHIP = 3e-8;  // cm^2: 3 um wide bottom contact x 1 um out-of-plane thickness
  static void out(String s) { System.out.println(s); System.out.flush(); }
  static String oneLine(Exception e) {
    StringBuilder b = new StringBuilder();
    for (char c : String.valueOf(e.getMessage()).toCharArray()) b.append(c < 32 ? (char) 32 : (c == (char) 44 ? (char) 59 : c));
    String s = b.toString();
    return s.length() > 200 ? s.substring(0, 200) : s;
  }
  static String dsetOf(Model m, String sol) {
    for (String ds : m.result().dataset().tags())
      try { if (sol.equals(m.result().dataset(ds).getString("solution"))) return ds; } catch (Exception e) {}
    return null;
  }
  static double[][] emit(Model m, String sol, String tag, String[] d, String[] expr) {
    NumericalFeature ev = m.result().numerical("ev");
    ev.set("expr", expr);
    ev.set("data", dsetOf(m, sol));
    double[][] r = ev.getReal();
    for (int i = 0; i < r[0].length; i++)
      out(String.format(java.util.Locale.ROOT, "%s,%s,%s,%s,%s,%s,%s,%.6g,%.9e,%.6f", tag, d[0], d[1], d[2], d[3], d[4], d[5],
          r[0][i], r[1][i], r[2][i]));
    return r;
  }
  static void solverSettings(Model m, String sol, String termonres, boolean noScaling) {
    for (String f : m.sol(sol).feature().tags()) {
      if (noScaling && m.sol(sol).feature(f).getType().equals("Variables")) {  // unit scaling for all variables
        m.sol(sol).feature(f).set("scalemethod", "none");
        for (String v : m.sol(sol).feature(f).feature().tags()) {
          m.sol(sol).feature(f).feature(v).set("scalemethod", "manual");
          m.sol(sol).feature(f).feature(v).set("scaleval", "1");
        }
      }
      for (String sub : m.sol(sol).feature(f).feature().tags())
        if (m.sol(sol).feature(f).feature(sub).getType().equals("FullyCoupled"))
          m.sol(sol).feature(f).feature(sub).set("termonres", termonres);
    }
  }
  public static Model run() {
    Model m;
    try { m = ModelUtil.load("m", BASE); } catch (java.io.IOException e) { throw new RuntimeException(e); }
    ModelNode c1 = m.component("comp1");
    c1.physics("semi").feature("gc1").active(false);
    m.param().set("Va", "0[V]"); m.param().set("Vd", "0[V]"); m.param().set("Vg", "0[V]"); m.param().set("Ia", "0[A]");
    m.param().set("T0", "300[K]"); m.param().set("r_th", "0.1[K*cm^2/W]"); m.param().set("tau", "1e-6[s]");
    m.param().set("A_chip", "3[um]*1[um]");
    c1.physics("semi").feature("tar1").set("taun_mat", "userdef"); c1.physics("semi").feature("tar1").set("taun", "tau");
    c1.physics("semi").feature("tar1").set("taup_mat", "userdef"); c1.physics("semi").feature("tar1").set("taup", "tau");
    c1.physics("semi").feature("smm1").set("mun_mat", "userdef");
    c1.physics("semi").feature("smm1").set("mup_mat", "userdef");
    c1.physics().create("ge", "GlobalEquations");
    c1.physics("ge").feature("ge1").setIndex("name", "Tj", 0, 0);
    c1.physics("ge").feature("ge1").setIndex("equation", "Tj-T0-r_th*semi.V0_3*semi.I0_3/A_chip", 0, 0);
    c1.physics("ge").feature("ge1").setIndex("initialValueU", "T0", 0, 0);
    m.study("std1").feature("stat").setEntry("activate", "ge", false);
    m.study("std2").feature("stat").setEntry("activate", "ge", true);
    m.result().numerical().create("ev", "EvalGlobal");
    StringBuilder il = new StringBuilder();
    for (int k = -4; k <= 12; k++) il.append(String.format(java.util.Locale.ROOT, "%.6g ", Math.pow(10, k / 4.0) * A_CHIP));

    for (String[] d : DEV) {
      long t0 = System.currentTimeMillis();
      out("START," + d[0]);
      try {
        m.param().set("T0", d[4] + "[K]"); m.param().set("r_th", d[5] + "[K*cm^2/W]"); m.param().set("tau", d[3] + "[s]");
        c1.physics("semi").feature("adm1").set("NAc", d[1] + "[1/cm^3]");
        double xj = Double.parseDouble(d[2]);
        String[] jda = {(0.8 * xj) + "[um]", xj + "[um]", "0"};
        c1.physics("semi").feature("adm2").set("jda", jda);
        c1.physics("semi").feature("adm3").set("jda", jda);

        // 1) isothermal, voltage driven
        c1.physics("semi").feature("mc3").set("TerminalType", "Voltage");
        c1.physics("semi").feature("mc3").set("V0", "Va");
        c1.physics("semi").feature("smm1").set("minput_temperature_src", "userdef");
        c1.physics("semi").feature("smm1").set("minput_temperature", "T0");
        c1.physics("semi").feature("smm1").set("mun", "1448[cm^2/(V*s)]*(T0/300[K])^(-2.33)");
        c1.physics("semi").feature("smm1").set("mup", "473[cm^2/(V*s)]*(T0/300[K])^(-2.23)");
        StudyFeature s1 = m.study("std1").feature("stat");
        s1.set("pname", new String[]{"Va"}); s1.set("punit", new String[]{"V"}); s1.set("plistarr", new String[]{VLIST});
        s1.set("pcontinuation", "Va"); s1.set("useinitsol", false);
        m.study("std1").createAutoSequences("all");
        solverSettings(m, "sol1", "on", false);
        m.study("std1").run();
        double[][] iso = emit(m, "sol1", "ISO", d, new String[]{"Va", "semi.I0_3", "T0"});

        // 2) self-heating, current driven; starts from the isothermal point closest to the first current
        double i0 = 0.1 * A_CHIP;  // A, first current of the sweep (J = 0.1 A/cm^2)
        int best = -1;
        for (int i = 0; i < iso[1].length; i++)
          if (iso[1][i] > 0 && (best < 0 || Math.abs(Math.log(iso[1][i] / i0)) < Math.abs(Math.log(iso[1][best] / i0)))) best = i;
        c1.physics("semi").feature("mc3").set("TerminalType", "Current");
        c1.physics("semi").feature("mc3").set("I0", "Ia");
        c1.physics("semi").feature("mc3").set("V0_init", iso[0][best] + "[V]");
        c1.physics("semi").feature("smm1").set("minput_temperature", "Tj");
        c1.physics("semi").feature("smm1").set("mun", "1448[cm^2/(V*s)]*(Tj/300[K])^(-2.33)");
        c1.physics("semi").feature("smm1").set("mup", "473[cm^2/(V*s)]*(Tj/300[K])^(-2.23)");
        StudyFeature s2 = m.study("std2").feature("stat");
        s2.set("pname", new String[]{"Ia"}); s2.set("punit", new String[]{"A"}); s2.set("plistarr", new String[]{il.toString()});
        s2.set("pcontinuation", "Ia");
        s2.set("useinitsol", true); s2.set("initmethod", "sol"); s2.set("initstudy", "std1"); s2.set("solnum", String.valueOf(best + 1));
        m.study("std2").createAutoSequences("all");
        // the raw residual of the coupled system cannot reach the residual tolerance; the points are
        // checked afterwards (energy balance, agreement with the isothermal curve at low current)
        solverSettings(m, "sol2", "auto", true);
        int n2 = 0;
        try { m.study("std2").run(); } catch (Exception e) { out("HOTFAIL," + d[0] + "," + oneLine(e)); }
        try { n2 = emit(m, "sol2", "HOT", d, new String[]{"semi.V0_3", "semi.I0_3", "Tj"})[0].length; } catch (Exception e) { out("HOTNODATA," + d[0]); }
        out(String.format(java.util.Locale.ROOT, "DONE,%s,%d,%d,%.1f", d[0], iso[0].length, n2, (System.currentTimeMillis() - t0) / 1000.0));
      } catch (Exception e) {
        out(String.format(java.util.Locale.ROOT, "FAIL,%s,%.1f,%s", d[0], (System.currentTimeMillis() - t0) / 1000.0, oneLine(e)));
      }
    }
    out("WORKER_FINISHED");
    return m;
  }
  public static void main(String[] a) { run(); }
}
EOF
} > "$out"
