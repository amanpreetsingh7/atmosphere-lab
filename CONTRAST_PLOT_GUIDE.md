# Atmosphere-based contrast versus separation

The reference graph demonstrates a useful plot format. Atmosphere Lab generates that format from your selected stellar and planetary atmosphere models, physical inputs and observing response. It does not adopt the reference target or its illustrative spectra.

1. Select the stellar and planetary atmosphere files or catalog entries in the sidebar. Set radii and common distance. Click **Run comparison**.
2. In **Filters & limits**, calculate a contrast using your throughput file and its response convention. Alternatively, supply a response directly in the separation tab.
3. Open **Contrast vs separation**. Choose **Selected planet atmosphere** or **Catalog temperature sequence**.
4. For a sequence, select available temperature nodes and enter the planet radii you want to compare. The sequence holds gravity, composition and other non-temperature parameters fixed.
5. Set the angular separation range and target label. Click **Build separation plot**.
6. Optionally upload an observed sensitivity curve. CSV columns are `separation_arcsec,contrast`, plus optional `pa_deg` for individual position-angle curves. An IDL `.sav` with `curves` and `separations` is also supported; `pas` and `threshold` are retained when present.
7. Click **Prepare separation export** to download PNG, PDF or a reproducibility ZIP.

At fixed temperature, model thermal-emission contrast is horizontal with separation. Selecting multiple temperatures and radii gives multiple atmosphere predictions; measured sensitivity may vary strongly with separation.

To examine stellar heating, check **Apply approximate stellar heating**. This requires a local temperature sequence that brackets the heated temperatures. The star's temperature comes from its actual header or declaration. No analytic spectrum replaces a missing model.

The approximation combines intrinsic and absorbed power, assuming full redistribution. It then interpolates integrated band signals at fixed non-temperature parameters. Unsupported temperatures leave gaps. A non-irradiated grid at a higher effective temperature is an approximation to an irradiated planet, not a solution for its irradiated atmosphere.

Projected separation in au is angular separation in arcsec times distance in pc. The true/projected ratio specifies an assumed geometry for the heating calculation. Neither orbital distance nor orbital phase is inferred from projected separation alone.

The optional mJy axis uses either the selected stellar atmosphere or a supplied stellar flux density with a compatible response and calibration convention. Its conversion is contrast times stellar Jy times 1000. A model/threshold crossing is not a statistical detection probability or a mass limit.

The imported summary is the arithmetic mean of finite saved curves. Missing values and incomplete position-angle runs are preserved. The program does not reconstruct measured sensitivity from a source-code file alone.

Native grids and responses need validation on your laptop. Native Bobcat mass/age mapping is available in Physical properties. Self-consistent irradiation is outside this release.

## Quick test inside the app

1. Select and run your atmosphere comparison.
2. If needed, use **Model library -> Add Bobcat temperature sequence**, then refresh model selectors and rerun the comparison.
3. Open **Contrast vs separation** and click **Set up an atmosphere plot test**.
4. Click **Build separation plot**. Your selected star is retained. Blue curves use real installed atmospheric spectra with approximate heating. Gray/black curves are simulated for the layout test and prominently labeled.
5. Click **Check this plot calculation** for numerical identity checks.
6. Click **Prepare separation export**, or use the complete combined ZIP in **Export & provenance**.

For science use, select the actual observing response and **Upload observed sensitivity**. Physical-property conversions and evolutionary table controls are under **Physical properties**. An evaluated mass/age planet can be selected in the separation tab, with complete-cell temperature/gravity interpolation and known composition compatibility checks.
