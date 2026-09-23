ACCESSION ?= ENCFF523NFH
OUTPUT ?= ENCSR000BRD_FOXA1_conservative_IDR_GRCh38.bed
URL = https://www.encodeproject.org/files/$(ACCESSION)/@@download/$(ACCESSION).bed.gz

.PHONY: all clean

all: $(OUTPUT)

$(OUTPUT):
	curl --fail --location --retry 3 --output $@.gz.tmp "$(URL)"
	gzip --test $@.gz.tmp
	gzip --decompress --stdout $@.gz.tmp > $@.tmp
	mv $@.tmp $@
	rm -f $@.gz.tmp

clean:
	rm -f "$(OUTPUT)" "$(OUTPUT).tmp" "$(OUTPUT).gz.tmp"
