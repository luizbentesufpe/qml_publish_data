#!/bin/bash
# check_results.sh
# Uso: ./check_results.sh /home/luizf/Desktop/projeto_refact/refactor_project

BASE_DIR="${1:-/home/luizf/Desktop/projeto_refact/refactor_project}"

PUB_DIRS=$(find "$BASE_DIR" -maxdepth 1 -type d -name "publication_out_*" | sort)

for PUB in $PUB_DIRS; do
    DATASET=$(basename "$PUB" | sed 's/publication_out_//')
    echo ""
    echo "======================================================="
    echo "DATASET: $DATASET"
    echo "======================================================="

    for SCENARIO_DIR in "$PUB"/*/; do
        SCENARIO=$(basename "$SCENARIO_DIR")
        echo ""
        echo "  ── Cenário: $SCENARIO"

        for NQ_DIR in "$SCENARIO_DIR"nq*/; do
            [ -d "$NQ_DIR" ] || continue
            NQ=$(basename "$NQ_DIR")
            echo ""
            echo "    ── $NQ"

            echo "    LIMPO:"
            for SEED_DIR in "$NQ_DIR"seed*/; do
                [ -d "$SEED_DIR" ] || continue
                SEED=$(basename "$SEED_DIR")
                # pega o log mais recente com glob de timestamp
                LOG=$(ls "$SEED_DIR/logs/final_test"-*.log 2>/dev/null | sort | tail -1)
                if [ -f "$LOG" ]; then
                    LAST=$(tail -1 "$LOG")
                    echo "      $SEED | $LAST"
                else
                    echo "      $SEED | [SEM LOG]"
                fi
            done

            echo "    RUIDOSO:"
            for SEED_DIR in "$NQ_DIR"seed*/; do
                [ -d "$SEED_DIR" ] || continue
                SEED=$(basename "$SEED_DIR")
                LOG=$(ls "$SEED_DIR/noise/logs/final_test"-*.log 2>/dev/null | sort | tail -1)
                if [ -f "$LOG" ]; then
                    LAST=$(tail -1 "$LOG")
                    echo "      $SEED | $LAST"
                else
                    echo "      $SEED | [SEM LOG RUÍDO]"
                fi
            done

            echo "    ALPHA:"
            for SEED_DIR in "$NQ_DIR"seed*/; do
                [ -d "$SEED_DIR" ] || continue
                SEED=$(basename "$SEED_DIR")
                LOG=$(ls "$SEED_DIR/logs/enc_params"-*.log 2>/dev/null | sort | tail -1)
                if [ -f "$LOG" ]; then
                    LAST=$(grep "\[SAVED\]" "$LOG" | tail -1)
                    echo "      $SEED | $LAST"
                else
                    echo "      $SEED | [SEM ALPHA]"
                fi
            done

            echo "    WARNINGS:"
            WARNS=$(find "$NQ_DIR" -name "enc_params-*.log" \
                | xargs grep -l "\[WARN\]" 2>/dev/null)
            if [ -z "$WARNS" ]; then
                echo "      nenhum"
            else
                echo "$WARNS" | while read f; do
                    grep "\[WARN\]" "$f" | while read w; do
                        echo "      ⚠️  $(basename $(dirname $(dirname $f)))/$(basename $(dirname $f)): $w"
                    done
                done
            fi

        done
    done
done

echo ""
echo "======================================================="
echo "RESUMO SAVES"
echo "======================================================="
TOTAL_CLEAN=$(find "$BASE_DIR" -path "*/logs/enc_params_final.pt" \
    ! -path "*/noise/*" 2>/dev/null | wc -l)
TOTAL_NOISY=$(find "$BASE_DIR" -path "*/noise/logs/enc_params_final.pt" \
    2>/dev/null | wc -l)

echo "  enc_params_final.pt limpos:   $TOTAL_CLEAN"
echo "  enc_params_final.pt ruidosos: $TOTAL_NOISY"
echo ""
echo "  Esperado limpos:   $(find "$BASE_DIR" -type d -name "seed*" ! -path "*/noise/*" | wc -l)"
echo "  Esperado ruidosos: $(find "$BASE_DIR" -type d -name "noise" | wc -l)"