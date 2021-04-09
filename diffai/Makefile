#-------------------------------
# Entry points
all: execs
execs: diff-main diff-batch
#-------------------------------
# Aliases
DIFF-MAIN =./_build/default/main_diff/entry_diff.exe
DIFF-BATCH=./_build/default/batch_diff/batch_diff.exe
#-------------------------------
# Continuity/diff-mainrentiability
diff-main:  $(DIFF-MAIN)  # Main analyzer (single file)
diff-batch: $(DIFF-BATCH) # Batch analyzer (many files)
$(DIFF-MAIN):
	dune build $(DIFF-MAIN) ; cp $(DIFF-MAIN)  diff-main.exe
$(DIFF-BATCH):
	dune build $(DIFF-BATCH); cp $(DIFF-BATCH) diff-batch.exe
#-------------------------------
# Running batch
run-diff-examples: $(DIFF-BATCH)
	$(DIFF-BATCH)
#-------------------------------
# PHONY misc targets
.PHONY: $(SUPP-MAIN) $(SUPP-BATCH) $(DIFF-MAIN) $(DIFF-BATCH) \
	all execs \
	diff-main diff-batch \
	run-diff-examples \
	wc edit clean
wc:
	wc */*.ml*
edit:
	emacs --background-color=Black --foreground-color=White makefile &
clean: 
	rm -rf */*~ _build *.exe
#-------------------------------
