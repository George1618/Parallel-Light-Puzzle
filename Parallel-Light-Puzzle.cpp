/* Projeto feito por George Harrison de Almeida Mendes, para a disciplina de Computação de Alto Desempenho
do curso de Ciência da Computação na Universidade Federal do Ceará */

#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include <limits.h>
#include <chrono>

#include "mpi.h"
#include "omp.h"

#define FOUND 1
#define NOTFOUND 2
#define MEMORYLIMIT 3

typedef struct Node {
    int* state;
    int level;
    int blank;
    int distance;
    int move;
    struct Node* parent;
    struct Node* next;
} Node;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv); // inicia o uso do MPI

    static const int MPIROOT = 0;
    int rank, mpisize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

    int length = 0, size = 0; // número de linhas (e de colunas) do puzzle
    char* filename = argv[1];

    // lê o arquivo apenas para saber o tamanho de cada linha/coluna, e então calcular o número total
    if (rank == MPIROOT) { // apenas a raíz lê o arquivo como no serial
        std::cout << "Iniciando leitura rapida do arquivo do puzzle..." << std::endl;
        std::ifstream prepuzzle(filename, std::ios::in); // arquivo que tem o puzzle em seu estado inicial

        std::string line;
        while (std::getline(prepuzzle, line)) { if (line.size() > 0) length++; } // conta o número de linhas não-vazias
        prepuzzle.close();

        size = length * length; // tamanho total do puzzle (número de valores + 1)
        if (length <= 2) { // o programa só aceita puzzle de tamanho 3x3 em diante
            if (length == 0) {
                std::cout << "Puzzle esta vazio. Verifique se o arquivo esta vazio, ou se o caminho esta correto." << std::endl;
            }
            else { std::cout << "Puzzle " << length << "x" << length << " nao tem tamanho significante..." << std::endl; }
            return 0;
        }
        MPI_Bcast(&length, 1, MPI_INT, MPIROOT, MPI_COMM_WORLD); // compartilha o valor de length com os demais
    }
    else {
        MPI_Bcast(&length, 1, MPI_INT, MPIROOT, MPI_COMM_WORLD); // recebe da raíz o valor de length
        size = length * length; // calcula size
    }
    MPI_Barrier(MPI_COMM_WORLD);

    static const int L = length, S = size, DSL = S - L, LL = L - 1; // alias constantes para cálculos
    static const int UP = -L, DOWN = L, LEFT = -1, RIGHT = 1; // tamanho dos movimentos
    static const int BLANKVAL = S; // valor inteiro do espaço em branco para representar em um array de inteiros
    Node* root = new Node{ new int[size], 0, 0, NULL, NULL }; // cria o estado inicial como a raíz
    
    // lê o estado inicial número por número e salva na raíz
    if (rank == MPIROOT) {
        std::cout << "Iniciando leitura do arquivo do puzzle..." << std::endl;
        std::ifstream puzzle(filename, std::ios::in); // reabre o arquivo, dessa vez para pegar todo o estado
        bool invalid = false;
        char digit;
        std::string num = "";
        int value = 0, ppos = 0;
        while (puzzle.get(digit) && !invalid) { // lê cada número do arquivo e salva em cada posição ppos de 0 a size-1
            if (digit != ',' && digit != '\n') { num += digit; } // acumula cada caractere até um dos delimitadores
            else {
                if (num == " ") {
                    value = BLANKVAL;  // guarda o espaço em branco como inteiro, sendo o size, que está além do intervalo [0, size-1]
                    root->blank = ppos; // guarda onde está o branco (economiza O(n2))
                }
                else {
                    try {
                        value = std::stoi(num);  // guarda o número como o inteiro que ele é; se não, um erro será lançado
                        if (value <= 0 || value >= S) { throw ""; } // lança erro se o número não estiver no intervalo [1,size-1]
                        for (int i = 0; i < ppos; i++) { // lança erro se tiver número repetido
                            if (root->state[ppos] == value) { throw ""; }
                        }
                    }
                    catch (const std::exception&) {
                        invalid = true;
                        std::cout << "Erro ao ler estado do puzzle. Corrija o arquivo de input." << std::endl;
                    }
                }
                root->state[ppos] = value; // salva no estado inicial root
                num = ""; // limpa a string de acumulação de caractere para a próxima iteração
                if (ppos < S - 1) ppos++; // itera, desde que não passe do limite de size-1
            }
        }
        puzzle.close();
        if (invalid) { // cancela para todos; arquivo não tem um puzzle válido
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_FILE); MPI_Finalize(); return -1;
        }
        MPI_Bcast(root->state, S, MPI_INT, MPIROOT, MPI_COMM_WORLD);
    }
    else {
        int* rootstate = new int[S];
        MPI_Bcast(&rootstate, S, MPI_INT, MPIROOT, MPI_COMM_WORLD);
        for (int i = 0; i < S; i++) {
            if (rootstate[i] == BLANKVAL) root->blank = i;
            root->state[i] = rootstate[i];
        }
        delete[] rootstate;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // algoritmo BnB
    std::cout << "Iniciando algoritmo..." << std::endl;
    Node* current = root; // a cada visita o nó atual mudará
    Node* best = NULL; // folha do melhor caminho

    int UB = 0, LB = 0; // limites superior (Upper Bound) e inferior (Lower Bound)
    int RESULT = 0; // guarda um dos casos de solução (definidos nas macros no início)

    const auto start = std::chrono::steady_clock::now(); // marcador de tempo de início do algoritmo a partir dos passos de laço

    // 1º passo: com a raíz; fora do laço por ter casos especiais
    int distance = 0, currBlank = current->blank, j, val; // heurística, posição do branco, da peça de troca, e valor na posição

    for (int i = 0; i < S; i++) { // calcula a heurística para o nó raíz
        if (i != currBlank) { // se não é o espaço em branco
            val = current->state[i] - 1; // enumera de 0 a (n2-1)-1 para o módulo ser de 0 a length-1
            distance += abs((val % L) - (i % L)) + abs((val / L) - (i / L));
        }
    }
    LB = current->distance = distance; // Único Lower Bound obtido neste algoritmo
    UB = LB + (LB / 4); // estimativa; uma heurística melhor é necessária
    if (distance == 0) { // primeiro estado já está solucionado
        best = current; LB = UB = current->level; RESULT = FOUND;
    }
    else { // raíz não-solucionada sempre é promissora; cria os filhos da raíz (até 4)
        int nextLevel = 1; // current->level + 1 = 0 + 1 = 1
        // criação dos filhos, verificando quais ramificações são possíveis baseando-se nos movimentos legais
        int nChildren = 0; // conta o número de filhos
        int nextBlank, nextValue, delta, nextDistance; // calcula a heurística para ser usada na ordenação dos filhos (OTIMIZAÇÃO)
        if (currBlank < DSL) { // pode ser movida para BAIXO
            try {
                nextBlank = currBlank + DOWN;
                nextValue = current->state[nextBlank] - 1;
                // diferença das distâncias de linhas na mudança (antes de mover - depois de mover)
                delta = abs((nextBlank / L) - (nextValue / L)) - abs((currBlank / L) - (nextValue / L));
                // com delta < 0, a distância aumentou, logo -delta/abs(delta)=-(-1)=+1; com > 0, a variação fica -(+1) = -1.
                nextDistance = distance - (delta / abs(delta));

                current->next = new Node{ NULL, nextLevel, nextBlank, nextDistance, DOWN, current, current->next };
                nChildren++;
            }
            catch (const std::exception&) {
                RESULT = MEMORYLIMIT;
            }
        }
        if ((currBlank % L) != LL) { // pode ser movida para a DIREITA
            try {
                nextBlank = currBlank + RIGHT;
                nextValue = current->state[nextBlank] - 1;
                // diferença das distâncias de colunas na mudança (antes de mover - depois de mover)
                delta = abs((nextBlank % L) - (nextValue % L)) - abs((currBlank % L) - (nextValue % L));
                nextDistance = distance - (delta / abs(delta));

                current->next = new Node{ NULL, nextLevel, nextBlank, nextDistance, RIGHT, current, current->next };
                nChildren++;

                // primeira possível decisão de ordem (máx. 2 filhos)
                if (nChildren == 2 && current->next->next->distance < nextDistance) { // tem os 2 filhos (BAIXO e DIREITA)
                    Node* CRIGHT = current->next, * CDOWN = current->next->next;
                    CRIGHT->next = CDOWN->next; CDOWN->next = CRIGHT; current->next = CDOWN; // reordena para -> BX. -> DIR.
                } // else: não tem outro filho ou este já é o melhor; não precisa reordenar
            }
            catch (const std::exception&) {
                RESULT = MEMORYLIMIT;
            }
        }
        if (currBlank >= L) { // pode ser movida para CIMA
            try {
                nextBlank = currBlank + UP;
                nextValue = current->state[nextBlank] - 1;
                // diferença das distâncias de linhas na mudança (antes de mover - depois de mover)
                delta = abs((nextBlank / L) - (nextValue / L)) - abs((currBlank / L) - (nextValue / L));
                nextDistance = distance - (delta / abs(delta));

                current->next = new Node{ NULL, nextLevel, nextBlank, nextDistance, UP, current, current->next };
                nChildren++;

                // segunda possível decisão de ordem (máx. 3 filhos)
                if (nChildren == 3) { // tem os 3 filhos (CIMA, BAIXO e DIREITA)
                    Node* CUP = current->next, * CPREV1 = CUP->next, * CPREV2 = CPREV1->next;
                    // ordena os três sem precisar saber quem é cada anterior
                    if (CPREV2->distance < nextDistance) { // CIMA tem que virar o último
                        CUP->next = CPREV2->next; CPREV2->next = CUP; current->next = CPREV1;
                    }
                    else if (CPREV1->distance < nextDistance) { // CIMA tem que estar no meio
                        CUP->next = CPREV2; CPREV1->next = CUP; current->next = CPREV1;
                    } // else: PREV1 < PREV2 garantido antes, então CIMA < ANT1 < ANT2 (como já está)
                }
                else if (nChildren == 2) { // tem este e exatamente 1 dos anteriores (ANTERIOR = BAIXO ou DIREITA)
                    Node* CUP = current->next, * CPREV = CUP->next;
                    // a distância já é trocada para estar no distance1
                    if (CPREV->distance < nextDistance) { // ANTERIOR é melhor; troca para -> ANT. -> CIMA
                        CUP->next = CPREV->next; CPREV->next = CUP; current->next = CPREV;
                    } // else: ANTERIOR não é melhor, este é: não precisa reordenar
                } // else: não tem outros filhos; não há o que reordernar
            }
            catch (const std::exception&) {
                RESULT = MEMORYLIMIT;
            }
        }
        if ((currBlank % L) != 0) { // pode ser movida para a ESQUERDA
            try {
                nextBlank = currBlank + LEFT;
                nextValue = current->state[nextBlank] - 1;
                // diferença das distâncias de colunas na mudança (antes de mover - depois de mover)
                delta = abs((nextBlank % L) - (nextValue % L)) - abs((currBlank % L) - (nextValue % L));
                nextDistance = distance - (delta / abs(delta));

                current->next = new Node{ NULL, nextLevel, nextBlank, nextDistance, LEFT, current, current->next };
                nChildren++;

                //  terceira possível decisão de ordem (máx. 4 filhos, e apenas 3 depois que há movimento contrário cortado)
                if (nChildren == 4) { // os quatro (este e todos os 3 acima) existem (isso ocorre apenas na raíz)
                    Node* CLEFT = current->next, * CPREV1 = CLEFT->next, * CPREV2 = CPREV1->next, * CPREV3 = CPREV2->next;
                    if (CPREV3->distance < nextDistance) { // quarto pior
                        CLEFT->next = CPREV3->next; CPREV3->next = CLEFT; current->next = CPREV1;
                    }
                    else if (CPREV2->distance < nextDistance) { // terceiro pior
                        CLEFT->next = CPREV3; CPREV2->next = CLEFT; current->next = CPREV1;
                    }
                    else if (CPREV1->distance < nextDistance) { // segundo pior
                        CLEFT->next = CPREV2; CPREV1->next = CLEFT; current->next = CPREV1;
                    }
                }
                else if (nChildren == 3) { // três filhos existem (este e outros 2, não importando quais são)
                    Node* CLEFT = current->next, * CPREV1 = CLEFT->next, * CPREV2 = CPREV1->next;
                    if (CPREV2->distance < nextDistance) {
                        CLEFT->next = CPREV2->next; CPREV2->next = CLEFT; current->next = CPREV1;
                    }
                    else if (CPREV1->distance < nextDistance) {
                        CLEFT->next = CPREV2; CPREV1->next = CLEFT; current->next = CPREV1;
                    }
                }
                else if (nChildren == 2) { // dois filhos existem (este e algum outro)
                    Node* CLEFT = current->next, * CPREV = CLEFT->next;
                    if (CPREV->distance < nextDistance) { // ANTERIOR é melhor; troca para -> ANT. -> CIMA
                        CLEFT->next = CPREV->next; CPREV->next = CLEFT; current->next = CPREV;
                    } // else ANTERIOR não é melhor, este é; não reordena
                }
                // else: filho único; não há troca
            }
            catch (const std::exception&) {
                RESULT = MEMORYLIMIT;
            }
        }
        // vai para o próximo nó
        current = current->next;
    }


    while (RESULT == 0) { // laço de visita
        if (current == NULL) {  // pai e irmãos não-nulos sempre são visitados antes (SERIAL), então não há mais como encontrar solução
            if (best == NULL) RESULT = NOTFOUND;
            else RESULT = FOUND; // caso não tenha fechado LB=UB mas acabado de visitar todos os nós
        }
        else { // Nó existe e não foi visitado
            int distance = current->distance, currBlank = current->blank, prevBlank = current->parent->blank;

            current->state = new int[S]; // como não foi inicializado pelo pai, inicializa aqui
            for (int i = 0; i < S; i++) { // copia o estado pai já fazendo o movimento das duas peças para o estado atual O(n2)
                if (i == currBlank) { j = prevBlank; }
                else if (i == prevBlank) { j = currBlank; }
                else { j = i; }
                current->state[i] = current->parent->state[j];
            }

            if (distance == 0) {// o estado está organizado: uma solução foi encontrada
                if (best == NULL || current->level < best->level) { // o atual é o melhor até agora ou é melhor que o de antes
                    best = current;
                    UB = best->level; // uma solução melhor é a que tiver nível menor que o atual, sendo este, portanto, um UB.
                    if (UB == LB) { // esta É uma solução ótima; não é necessário procurar outra
                        RESULT = FOUND;
                    }
                    else { // pode haver ainda melhores nós; continua com os próximos nós
                        current = current->next;
                    }
                }
                else { // este nó é apenas um de igual nível; continua procurando nos próximos nós
                    current = current->next;
                }
            }
            else if (current->level + distance < UB) { // estado atual tem filhos promissores
                int nextLevel = current->level + 1;
                // criação dos filhos, verificando quais ramificações são possíveis baseando-se nos movimentos legais
                int nChildren = 0;
                int nextBlank, nextValue, nextDistance, delta; // calcula a heurística para ser usada na ordenação dos filhos
                if (currBlank < DSL && current->move != UP) { // pode ser movida para BAIXO
                    try {
                        nextBlank = currBlank + DOWN;
                        nextValue = current->state[nextBlank] - 1;
                        // diferença das distâncias de linhas na mudança (antes de mover - depois de mover)
                        delta = abs((nextBlank / L) - (nextValue / L)) - abs((currBlank / L) - (nextValue / L));
                        // com delta < 0, a distância aumentou, logo -delta/abs(delta)=-(-1)=+1; com > 0, a variação fica -(+1) = -1.
                        nextDistance = distance - (delta / abs(delta));

                        current->next = new Node{ NULL, nextLevel, nextBlank, nextDistance, DOWN, current, current->next };
                        nChildren++;
                    }
                    catch (const std::exception&) {
                        RESULT = MEMORYLIMIT; break;
                    }
                }
                if ((currBlank % L) != LL && current->move != LEFT) { // pode ser movida para a DIREITA
                    try {
                        nextBlank = currBlank + RIGHT;
                        nextValue = current->state[nextBlank] - 1;
                        // diferença das distâncias de colunas na mudança (antes de mover - depois de mover)
                        delta = abs((nextBlank % L) - (nextValue % L)) - abs((currBlank % L) - (nextValue % L));
                        nextDistance = distance - (delta / abs(delta));

                        current->next = new Node{ NULL, nextLevel, nextBlank, nextDistance, RIGHT, current, current->next };
                        nChildren++;

                        // primeira possível decisão de ordem (máx. 2 filhos)
                        if (nChildren == 2 && current->next->next->distance < nextDistance) { // tem os 2 filhos (BAIXO e DIREITA)
                            Node* CRIGHT = current->next, * CDOWN = current->next->next;
                            CRIGHT->next = CDOWN->next; CDOWN->next = CRIGHT; current->next = CDOWN; // troca para -> BX. -> DIR.
                        } // else não tem outro filho ou o atual DIREITA é o melhor (já está na ordem DIR. -> BX.)
                    }
                    catch (const std::exception&) {
                        RESULT = MEMORYLIMIT; break;
                    }
                }
                if (currBlank >= L && current->move != DOWN) { // pode ser movida para CIMA
                    try {
                        nextBlank = currBlank + UP;
                        nextValue = current->state[nextBlank] - 1;
                        // diferença das distâncias de linhas na mudança (antes de mover - depois de mover)
                        delta = abs((nextBlank / L) - (nextValue / L)) - abs((currBlank / L) - (nextValue / L));
                        nextDistance = distance - (delta / abs(delta));

                        current->next = new Node{ NULL, nextLevel, nextBlank, nextDistance, UP, current, current->next };
                        nChildren++;

                        // segunda possível decisão de ordem (máx. 3 filhos)
                        if (nChildren == 3) { // tem os 3 filhos (CIMA, BAIXO e DIREITA)
                            Node* CUP = current->next, * CPREV1 = CUP->next, * CPREV2 = CPREV1->next;
                            // ordena os três sem precisar saber quem é cada anterior
                            if (CPREV2->distance < nextDistance) { // CIMA tem que virar o último
                                CUP->next = CPREV2->next; CPREV2->next = CUP; current->next = CPREV1;
                            }
                            else if (CPREV1->distance < nextDistance) { // CIMA tem que estar no meio
                                CUP->next = CPREV2; CPREV1->next = CUP; current->next = CPREV1;
                            } // else: CIMA < ANT1 < ANT2 (como já está)
                        }
                        else if (nChildren == 2) { // tem este e exatamente 1 dos anteriores (ANTERIOR = BAIXO ou DIREITA)
                            Node* CUP = current->next, * CPREV = CUP->next;
                            if (CPREV->distance < nextDistance) { // ANTERIOR é melhor; troca para -> ANT. -> CIMA
                                CUP->next = CPREV->next; CPREV->next = CUP; current->next = CPREV;
                            } // else: ANTERIOR não é melhor, este é; não precisa de reordenação
                        } // else: filho único; não há o que reordenação
                    }
                    catch (const std::exception&) {
                        RESULT = MEMORYLIMIT; break;
                    }
                }
                if ((currBlank % L) != 0 && current->move != RIGHT) { // pode ser movida para a ESQUERDA
                    try {
                        nextBlank = currBlank + LEFT;
                        nextValue = current->state[nextBlank] - 1;
                        // diferença das distâncias de colunas na mudança (antes de mover - depois de mover)
                        delta = abs((nextBlank % L) - (nextValue % L)) - abs((currBlank % L) - (nextValue % L));
                        nextDistance = distance - (delta / abs(delta));

                        current->next = new Node{ NULL, nextLevel, nextBlank, nextDistance, LEFT, current, current->next };
                        nChildren++;

                        //  terceira possível decisão de ordem (máx. 3 filhos, pois 1 dos filhos sempre é cortado pelo move check)
                        if (nChildren == 3) { // três filhos existem (este e outros 2, não importando quais são)
                            Node* CLEFT = current->next, * CPREV1 = CLEFT->next, * CPREV2 = CPREV1->next;
                            if (CPREV2->distance < nextDistance) {
                                CLEFT->next = CPREV2->next; CPREV2->next = CLEFT; current->next = CPREV1;
                            }
                            else if (CPREV1->distance < nextDistance) {
                                CLEFT->next = CPREV2; CPREV1->next = CLEFT; current->next = CPREV1;
                            } // else: mantém como está
                        }
                        else if (nChildren == 2) { // dois filhos existem (este e algum outro)
                            Node* CLEFT = current->next, * CPREV = CLEFT->next;
                            if (CPREV->distance < nextDistance) { // ANTERIOR é melhor; troca para -> ANT. -> CIMA
                                CLEFT->next = CPREV->next; CPREV->next = CLEFT; current->next = CPREV;
                            } // else ANTERIOR não é melhor, este é; não reordena
                        } // else: filho único; não troca porque não há mais cálculo com as distâncias
                    }
                    catch (const std::exception&) {
                        RESULT = MEMORYLIMIT; break;
                    }
                }
                current = current->next; // vai para o próximo nó
            }
            else { // (level+distance>=UB): estado atual não é solução ótima nem tem filhos promissores (que possam ser ótimos) 
                current = current->next; // então "poda": vai direto ao próximo nó
            }
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const auto time = end - start;
    std::cout << "Fim do algoritmo! (" << time.count() << " nanossegundos)" << std::endl; // fim do BnB

    if (RESULT == MEMORYLIMIT) return 1;

    // salva a execução do algoritmo em um arquivo
    std::cout << "Salvando resultados do algoritmo no arquivo de saida..." << std::endl;
    if (rank == 0 && argc >= 3) { // se um arquivo de saída for fornecido (Apenas o rank 0 escreve)
        char* outputname = argv[2];
        std::ofstream puzzleRes(outputname, std::ofstream::trunc); // modo de leitura: limpa e escreve

        std::string result;  // salva o tipo de resultado na primeira linha
        switch (RESULT) {
        case FOUND: result = "Solução encontrada"; break;
        case NOTFOUND: result = "Solução não encontrada"; break;
        case MEMORYLIMIT: result = "Não foi possível encontrar solução antes da memória esgotar"; break;
        default: result = "Resultado inesperado";  break;
        }
        puzzleRes << result << "\n";
        puzzleRes << "LB: " << LB << " ; UB: " << UB << "\n";

        // repete a escrita para a árvore inteira, a partir da raíz
        Node* node = root;
        while (node != NULL) {
            int resValue; // guarda cada número do puzzle em cada iteração
            if (node->state != NULL) for (int i = 0; i < L; i++) {
                int row = i * L;
                for (int j = 0; j < L; j++) {
                    resValue = node->state[row + j];
                    if (resValue == size) { puzzleRes << " "; } // imprime o espaço vazio (representado por size)
                    else { puzzleRes << resValue; } // imprime o número da posição atual
                    if (j < LL) puzzleRes << ",";
                }
                puzzleRes << "\n";
            }
            else { // o puzzle filho não foi instanciado porque nem precisou
                puzzleRes << "[NULL]\n";
            }
            puzzleRes << "c:" << node;
            if (node == best) puzzleRes << "*";
            if (node == current) puzzleRes << "!";
            puzzleRes << "\nl:" << node->level;
            puzzleRes << "\nb:" << node->blank + 1;
            puzzleRes << "\nd:" << node->distance << "\n";
            if (node->move == 0) puzzleRes << "START";
            else if (node->move == L) puzzleRes << "DOWN";
            else if (node->move == 1) puzzleRes << "RIGHT";
            else if (node->move == -L) puzzleRes << "UP";
            else if (node->move == -1) puzzleRes << "LEFT";
            puzzleRes << "\np:" << node->parent;
            puzzleRes << "\nn:" << node->next << "\n";
            node = node->next;
        }
        // fecha o arquivo de saída
        puzzleRes.close();
        std::cout << "Fim da escrita no arquivo de saida." << std::endl;
    }
    if (rank == 0 && argc >= 4 && best != NULL) { // se um segundo arquivo de saída for fornecido, para informações do melhor nó:
        char* outputname2 = argv[3];
        std::ofstream puzzleBest(outputname2, std::ofstream::trunc);

        puzzleBest << "Caminho Inverso do Melhor à Raíz" << "\n";
        puzzleBest << "LB: " << LB << " ; UB: " << UB << "\n";

        // repete a escrita para a árvore inteira, a partir da raíz
        Node* node = best;
        while (node != NULL) {
            int resValue; // guarda cada número do puzzle em cada iteração
            if (node->state != NULL) for (int i = 0; i < L; i++) {
                int row = i * L;
                for (int j = 0; j < L; j++) {
                    resValue = node->state[row + j];
                    if (resValue == size) { puzzleBest << " "; } // imprime o espaço vazio (representado por size)
                    else { puzzleBest << resValue; } // imprime o número da posição atual
                    if (j < LL) puzzleBest << ",";
                }
                puzzleBest << "\n";
            }
            else { // o nó não foi instanciado 
                puzzleBest << "[NULL]\n";
            }
            puzzleBest << "c:" << node;
            if (node == best) puzzleBest << "*";
            puzzleBest << "\nl:" << node->level;
            puzzleBest << "\nb:" << node->blank + 1;
            puzzleBest << "\nd:" << node->distance << "\n";
            if (node->move == 0) puzzleBest << "START";
            else if (node->move == L) puzzleBest << "DOWN";
            else if (node->move == 1) puzzleBest << "RIGHT";
            else if (node->move == -L) puzzleBest << "UP";
            else if (node->move == -1) puzzleBest << "LEFT";
            puzzleBest << "\np:" << node->parent;
            puzzleBest << "\nn:" << node->next << "\n\n";
            node = node->parent;
        }
        // fecha o arquivo de saída
        puzzleBest.close();
        std::cout << "Fim da escrita no segundo arquivo de saida." << std::endl;
    }

    std::cout << "Limpando memoria..." << std::endl;
    // limpa as alocações de memória feitas para a árvore de estados
    Node* temp = NULL, * deleting = root;
    while (deleting != NULL) { // navega o encadeamento da árvore
        temp = deleting;
        deleting = deleting->next; // temp = atual, deleting = próximo, para então apagar o atual (temp) sem perder quem é o próximo
        delete[] temp->state; // única alocação própria da estrutura é o vetor do estado 
        delete temp; // nós também são alocados com new
    }

    std::cout << "Fim do programa." << std::endl;

    MPI_Finalize(); // termina o uso do MPI
    return 0;
}
