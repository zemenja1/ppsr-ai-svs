// Template: run_generator.sh replaces %MPH% by the model path and %KMU% by the list of permeability scaling factors and compiles the class.
import com.comsol.model.*;
import com.comsol.model.util.*;
public class GenRun {
  static void out(String s) { System.out.println(s); System.out.flush(); }
  static String row(String tag, double[][] d) { StringBuilder sb = new StringBuilder(tag); for (double[] r : d) for (double v : r) sb.append(",").append(v); return sb.toString(); }
  public static void main(String[] a) {
    Model m;
    try { m = ModelUtil.load("gen", "%MPH%"); } catch (java.io.IOException e) { throw new RuntimeException(e); }
    m.label("generator_2d_yoke_sweep");
    var c = m.component("comp1");
    m.param().set("kmu", "1", "Permeability scaling of the iron B(H) curve");
    m.param().set("sig_l", "1e7[S/m]", "Conductivity of the laminated iron (loss model)");
    m.param().set("d_l", "0.35[mm]", "Lamination thickness (loss model)");
    var bh = c.material("mat2").propertyGroup("BHCurve");
    bh.set("normB", "BH(kmu*normHin)");
    bh.set("normH", "BH_inv(normBin)/kmu");
    bh.set("Wpm", "BH_prim(kmu*normHin)/kmu");
    var ls = c.physics("rmm").feature("al2").feature("loss1");
    ls.set("sigma", "sig_l");
    ls.set("d", "d_l");
    int[][] doms = {{2}, {28}, {29}, {20,21,22,23,24,25,26,27}};
    String[] dn = {"stator", "rotor", "shaft", "magnets"};
    for (int k = 0; k < 4; k++) {
      m.result().numerical().create("q" + k, "IntSurface");
      m.result().numerical("q" + k).set("data", "dset3");
      m.result().numerical("q" + k).selection().set(doms[k]);
      m.result().numerical("q" + k).set("expr", new String[]{"rmm.Qh*L"});
    }
    m.result().numerical().create("emf", "EvalGlobal");
    m.result().numerical("emf").set("data", "dset1");
    m.result().numerical("emf").set("expr", new String[]{"rmm.VCoil_1", "t"});
    String[] fe = {"rmm.normB", "rmm.normB", "rmm.normB/(mu0_const*rmm.normH)"};
    String[] ft = {"MaxSurface", "AvSurface", "AvSurface"};
    int[][] fd = {{2}, {2}, {2}};
    for (int k = 0; k < 3; k++) {
      m.result().numerical().create("f" + k, ft[k]);
      m.result().numerical("f" + k).set("data", "dset1");
      m.result().numerical("f" + k).selection().set(fd[k]);
      m.result().numerical("f" + k).set("expr", new String[]{fe[k]});
    }
    m.result().numerical().create("r0", "MaxSurface");
    m.result().numerical("r0").set("data", "dset1");
    m.result().numerical("r0").selection().set(new int[]{28});
    m.result().numerical("r0").set("expr", new String[]{"rmm.normB"});
    double[] kmu = {%KMU%};
    double[] sig = {0, 1e6, 2e6, 3e6, 5e6, 7.5e6, 1e7, 1.25e7};
    double[] dl = {0.1, 0.2, 0.35, 0.5, 1.0};
    for (double km : kmu) {
      try {
        m.param().set("kmu", "" + km);
        m.param().set("sig_l", "1e7[S/m]"); m.param().set("d_l", "0.35[mm]");
        long t0 = System.currentTimeMillis();
        m.study("std1").run();
        out("FIELD," + km + "," + (System.currentTimeMillis() - t0) / 1000.0);
        out(row("EMF," + km, m.result().numerical("emf").getReal()));
        out(row("BMAXS," + km, m.result().numerical("f0").getReal()));
        out(row("BAVS," + km, m.result().numerical("f1").getReal()));
        out(row("MURS," + km, m.result().numerical("f2").getReal()));
        out(row("BMAXR," + km, m.result().numerical("r0").getReal()));
        // loss variants: grid plus split helpers (sigma 0 -> hysteresis only; d tiny -> hysteresis + excess)
        java.util.List<double[]> var = new java.util.ArrayList<>();
        for (double s : sig) for (double d : dl) { if (s == 0 && d != 0.35) continue; var.add(new double[]{s, d}); }
        for (double s : sig) if (s > 0) var.add(new double[]{s, 1e-6});
        for (double[] v : var) {
          m.param().set("sig_l", v[0] + "[S/m]");
          m.param().set("d_l", v[1] + "[mm]");
          long t1 = System.currentTimeMillis();
          m.study("std2").run();
          StringBuilder sb = new StringBuilder("LOSS," + km + "," + v[0] + "," + v[1]);
          for (int k = 0; k < 4; k++) sb.append(",").append(m.result().numerical("q" + k).getReal()[0][0]);
          sb.append(",").append((System.currentTimeMillis() - t1) / 1000.0);
          out(sb.toString());
        }
      } catch (Exception ex) { out("FAIL," + km + "," + ex.getMessage().split("\n")[0]); }
    }
    out("DONE");
  }
}
