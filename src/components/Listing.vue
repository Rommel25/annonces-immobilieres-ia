<template>
  <div>
    <h3>Liste d'appartements</h3>
    <v-row>
      <!-- Colonne 1 : Tableau des appartements -->
      <v-col cols="12" md="8">
        <v-data-table
          :headers="headers"
          :items="suites"
          class="elevation-1"
        >
          <template v-slot:item.nbRooms="{ item }">
            <span>{{ item.nbRooms }}</span>
          </template>
          <template v-slot:item.surface="{ item }">
            <span>{{ item.surface }} m²</span>
          </template>
          <template v-slot:item.nbWindows="{ item }">
            <span>{{ item.nbWindows }}</span>
          </template>
          <template v-slot:item.annee="{ item }">
            <span>{{ item.annee }}</span>
          </template>
          <template v-slot:item.balcon="{ item }">
            <span>{{ item.balcon ? 'Oui' : 'Non' }}</span>
          </template>
          <template v-slot:item.garage="{ item }">
            <span>{{ item.garage ? 'Oui' : 'Non' }}</span>
          </template>
          <template v-slot:item.note="{ item }">
            <span>{{ item.note }}/5</span>
          </template>
          <template v-slot:item.price_category="{ item }">
            <span>{{ item.price_category }}</span>
          </template>
          <template v-slot:item.actions="{ item }">
            <v-btn color="red" @click="supprimerAppartement(item.id)">
              Supprimer
            </v-btn>
          </template>
        </v-data-table>
      </v-col>

      <v-col cols="12" md="4">
        <add @appartement-ajoute="ajouterAppartement" />
      </v-col>
    </v-row>
  </div>
</template>

<script>
import { defineComponent, reactive, onMounted } from 'vue';
import Add from './Add.vue';
import axios from 'axios';

export default defineComponent({
  components: {
    Add
  },
  setup() {
    const headers = [
      { text: 'Nombre de chambres', value: 'nbRooms' },
      { text: 'Surface (m²)', value: 'surface' },
      { text: 'Nombre de fenêtres', value: 'nbWindows' },
      { text: 'Année', value: 'annee' },  // Ajout de la colonne pour l'année
      { text: 'Balcon', value: 'balcon' },  // Ajout de la colonne pour le balcon
      { text: 'Garage', value: 'garage' },  // Ajout de la colonne pour le garage
      { text: 'Note', value: 'note' },  // Ajout de la colonne pour la note
      { text: 'Catégorie de prix', value: 'price_category' },  // Ajout de la colonne pour la catégorie de prix
      { text: 'Actions', value: 'actions', sortable: false } // Colonne pour les actions
    ];

    let suites = reactive([]);

    // Fonction pour charger les appartements depuis l'API
    const fetchAppartements = async () => {
      try {
        const response = await axios.get('http://localhost:3000/api/appartements');
        suites.push(...response.data); // Charger les données de l'API dans suites
      } catch (error) {
        console.error('Erreur lors du chargement des appartements :', error);
      }
    };

    // Charger les appartements au montage du composant
    onMounted(fetchAppartements);

    const ajouterAppartement = async (nouvelAppartement) => {
      try {
        // Envoyer l'appartement au serveur avec Axios
        const response = await axios.post('http://localhost:3000/api/appartements', {
          id: suites.length + 1,
          ...nouvelAppartement
        });

        // Mettre à jour la liste des appartements après l'ajout
        suites.push({ ...nouvelAppartement, id: suites.length + 1 });
      } catch (error) {
        console.error('Erreur lors de l\'ajout de l\'appartement :', error);
      }
    };

    const supprimerAppartement = async (id) => {
      try {
        // Supprimer l'appartement sur le serveur avec Axios
        await axios.delete(`http://localhost:3000/api/appartements/${id}`);
        
        // Mettre à jour la liste localement après suppression
        const index = suites.findIndex((suite) => suite.id === id);
        if (index !== -1) {
          suites.splice(index, 1);
        }
      } catch (error) {
        console.error('Erreur lors de la suppression de l\'appartement :', error);
      }
    };

    return { headers, suites, ajouterAppartement, supprimerAppartement };
  }
});
</script>
