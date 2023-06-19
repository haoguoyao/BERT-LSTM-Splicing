SPECIES = config["SPECIES"]

if SPECIES == "Human":
    REF = "/rhome/ghao004/bigdata/lstm_splicing/genome/GRCh38.primary_assembly.genome.fa"
    GTF = "/rhome/ghao004/bigdata/lstm_splicing/genome/gencode.v42.primary_assembly.annotation.gtf"
    CHROMS = "chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY"


# FILE_IDS = "K562 GM12878 HEPG2 HMEC".split()

# tmp = []
# for file in FILE_IDS:
#     if SPECIES in file:
#         tmp.append(file)
# FILE_IDS = tmp


FILE_IDS = ["GM12878"]
rule all:
     input:
         expand("bams/{id}.filtered.SpliSER.tsv", id=FILE_IDS),
         expand("{id}.rsem.genes.results", id=FILE_IDS)

# rule bam_to_fastq:
#     input:
#         "{id}.bam"
#     output:
#         "{id}.fastq.gz"
#     shell:
#         "samtools fastq {input} | gzip > {output}"


rule build_index:
    input:
        loc="STAR_%s" % SPECIES,
        fa=REF,
        gtf=GTF
        # path_to_STAR="/bigdata/lstm_splicing/STAR-2.7.10a/bin/Linux_x86_64/"
    output:
        "STAR_%s/SAindex" % SPECIES
    threads: 16
    shell:
        "/rhome/ghao004/bigdata/lstm_splicing/bin/STAR "
        "--runMode genomeGenerate "
        "--genomeDir {input.loc} "
        "--genomeFastaFiles {input.fa} "
        "--sjdbGTFfile {input.gtf} "
        "--runThreadN {threads} "
        "--limitGenomeGenerateRAM 50000000000"

rule first_pass:
     input:
         "STAR_%s/SAindex" % SPECIES,
         loc="STAR_%s" % SPECIES,
         # fq="{id}.fastq.gz",
         path_to_STAR="/rhome/ghao004/bigdata/lstm_splicing/bin"
     output:
         output_bam="bams/{id}.p1.Aligned.sortedByCoord.out.bam",
         output_SJ="bams/{id}.p1.SJ.out.tab"
     threads: 16
     shell:
         "{input.path_to_STAR}/STAR "
         "--genomeDir {input.loc} "
         "--readFilesIn /rhome/ghao004/bigdata/lstm_splicing/{wildcards.id}/ENCFF001REK.fastq.gz,/rhome/ghao004/bigdata/lstm_splicing/{wildcards.id}/ENCFF001REI.fastq.gz /rhome/ghao004/bigdata/lstm_splicing/{wildcards.id}/ENCFF001REJ.fastq.gz,/rhome/ghao004/bigdata/lstm_splicing/{wildcards.id}/ENCFF001REH.fastq.gz "
         "--readFilesCommand gunzip -c "
         "--outSAMtype BAM SortedByCoordinate "
         "--outFileNamePrefix bams/{wildcards.id}.p1. "
         "--runThreadN {threads} "
         "--genomeLoad LoadAndRemove "
         "--limitBAMsortRAM 50000000000"

rule merge_splice_junctions:
     input:
         sjs = expand("bams/{id}.p1.SJ.out.tab", id = FILE_IDS)
     output:
         sjs = "bams/SJ.out.p1_merged.%s.tab" % SPECIES
     shell:
         # Retain splice junctions with at least 3 uniquely mapped fragments per sample.
         "cat {input.sjs} | awk '$7 >= 3' | cut -f1-4 | sort -u > {output.sjs}"

rule second_pass:
     input:
         loc="STAR_%s" % SPECIES,
         # fq="{id}.fastq.gz",
         path_to_STAR="/rhome/ghao004/bigdata/lstm_splicing/bin",
         sjs="bams/SJ.out.p1_merged.%s.tab" % SPECIES
     output:
         output_bam="bams/{id}.p2.Aligned.sortedByCoord.out.bam",
         transcriptome_bam="bams/{id}.p2.Aligned.toTranscriptome.out.bam",
         output_SJ="bams/{id}.p2.SJ.out.tab"
     threads: 32
     shell:
         "{input.path_to_STAR}/STAR "
         "--genomeDir {input.loc} "
         "--readFilesIn /rhome/ghao004/bigdata/lstm_splicing/{wildcards.id}/ENCFF001REK.fastq.gz,/rhome/ghao004/bigdata/lstm_splicing/{wildcards.id}/ENCFF001REI.fastq.gz /rhome/ghao004/bigdata/lstm_splicing/{wildcards.id}/ENCFF001REJ.fastq.gz,/rhome/ghao004/bigdata/lstm_splicing/{wildcards.id}/ENCFF001REH.fastq.gz "
         "--readFilesCommand gunzip -c "
         "--sjdbFileChrStartEnd {input.sjs} "
         "--outSAMtype BAM SortedByCoordinate "
         "--outFileNamePrefix bams/{wildcards.id}.p2. "
         "--runThreadN {threads} "
         "--outFilterType BySJout "
         "--outFilterMultimapNmax 20 "
         "--alignSJoverhangMin 8 "
         "--alignSJDBoverhangMin 1 "
         "--outFilterMismatchNmax 999 "
         "--outFilterMismatchNoverReadLmax 0.04 "
         "--alignIntronMin 20 "
         "--alignIntronMax 1000000 "
         "--alignMatesGapMax 1000000 "
         "--quantMode TranscriptomeSAM "

rule rsem_index:
    input:
        fa=REF,
        gtf=GTF
    output:
        "RSEM_%s.idx.fa" % SPECIES
    params:
        "RSEM_%s" % SPECIES
    threads: 32
    shell:
        "../RSEM/rsem-prepare-reference -p 16 --gtf {input.gtf} {input.fa} {params}"

rule rsem:
    input:
        bam="bams/{id}.p2.Aligned.toTranscriptome.out.bam",
        loc="RSEM_%s.idx.fa" % SPECIES
    output:
        "{id}.rsem.genes.results"
    params:
        "RSEM_%s" % SPECIES
    threads: 32
    # --paired-end or not
    shell:
        "../RSEM/rsem-calculate-expression --paired-end --num-threads {threads} --alignments {input.bam} {params} {wildcards.id}.rsem"

# TruSeq
rule add_strand:
    input:
        bam="bams/{id}.p2.Aligned.sortedByCoord.out.bam",
        path="/rhome/ghao004/bigdata/lstm_splicing/STAR-2.7.10a/extras/scripts"
    output:
        "bams/{id}.p2.Aligned.sortedByCoord.XS.out.bam"
    shell:
        "samtools view -h {input.bam} | awk -v strType=2 -f {input.path}/tagXSstrandedData.awk | samtools view -bS - > {output}"

rule sort_by_read:
    input:
        "bams/{id}.p2.Aligned.sortedByCoord.XS.out.bam"
    output:
        "bams/{id}.p2.Aligned.sortedByRead.out.bam"
    threads: 8
    shell:
        "samtools sort -n -@ {threads} -o {output} {input}"

# assign multimapping reads
# uses XS tag to determine strand 
rule mmr:
    input:
        bam="bams/{id}.p2.Aligned.sortedByRead.out.bam",
        #gtf=GTF
    output:
        "bams/{id}.mmr.bam"
    threads: 32
    shell:
        "../mmr/mmr -S -o {output} -b -t {threads} {input.bam}"

rule sort_index:
    input:
        "bams/{id}.mmr.bam"
    output:
        bam="bams/{id}.mmr.sorted.bam",
        index="bams/{id}.mmr.sorted.bam.bai",
    threads: 8
    shell:
        """
        samtools sort {input} -o {output.bam} -@ {threads}
        samtools index {output.bam}
        """

# -s 0 => uses XS tag to determine strand
rule filter_index_and_calc_juncs:
    input:
        "bams/{id}.mmr.sorted.bam"
    output:
        juncs="bams/{id}.juncs.bed",
        bam="bams/{id}.mmr.sorted.filtered.bam",
        index="bams/{id}.mmr.sorted.filtered.bam.bai"
    params:
        CHROMS
    shell:
        """
        /rhome/ghao004/bigdata/lstm_splicing/sambamba-0.8.2-linux-amd64-static  slice -o {output.bam} {input} {params}
        samtools index {output.bam}
        regtools junctions extract -a 6 -o {output.juncs} -s XS {output.bam}
        """

rule calc_usage:
    input:
        bam="bams/{id}.mmr.sorted.filtered.bam",
        bed="bams/{id}.juncs.bed",
        annot=GTF
    output:
        "bams/{id}.filtered.SpliSER.tsv"
    params:
        "bams/{id}.filtered"
        
        
    
    shell:
        "python ../SpliSER/SpliSER_v0.1.7.py process --isStranded -s rf -B {input.bam} -o {params} -b {input.bed} -A {input.annot}"
# for stranded SPLISER, RSEM, regtool   
        
        