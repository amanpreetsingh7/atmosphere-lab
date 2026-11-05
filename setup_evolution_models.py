"""Install the small official Bobcat evolution archive on the user's laptop."""
from pathlib import Path
import json
from atmosphere_compare.downloads import download,extract_selected,digest_file
from atmosphere_compare.evolution import read_bobcat_tracks

ROOT=Path(__file__).resolve().parent
URL='https://zenodo.org/api/records/5063476/files/evolution_and_photometery.tar.gz/content'
MD5='a7c844bdcc4093e5b667ecf944ca2128'


def main():
    archive=download(URL,ROOT/'models/archives/evolution_and_photometery.tar.gz',950605,MD5)
    destinations={}
    for metal in ('-0.5','+0.0','+0.5'):
        name='nc%s_co1.0_mass'%metal
        destinations['evolution_tables/evo_tables%s/%s'%(metal,name)]=ROOT/'models/evolution'/name
    destinations['Evolution_and_Photometry_README.pdf']=ROOT/'models/evolution/Bobcat_README.pdf'
    members=extract_selected(archive,destinations)
    records=[]
    for metal in ('-0.5','+0.0','+0.5'):
        path=ROOT/'models/evolution'/('nc%s_co1.0_mass'%metal)
        rows,meta=read_bobcat_tracks(path.read_bytes(),path.name,'https://zenodo.org/records/5063476')
        meta.update(metallicity_dex=float(metal),co_relative_solar=1.,archive_md5=MD5)
        records.append({'path':str(path),'rows':len(rows),'metadata':meta})
        print('Installed evolution tracks:',path.name,';',len(rows),'mass/age rows',flush=True)
    (ROOT/'models/evolution/installation.json').write_text(json.dumps({'records':records,'members':members},indent=2))
    print('READY: select installed Bobcat tracks in Physical properties > Mass and age.',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:print('EVOLUTION INSTALL STOPPED:',exc,flush=True);raise SystemExit(1)
